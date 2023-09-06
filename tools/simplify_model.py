#!/usr/bin/env python
# coding: utf-8
import copy

import onnx
from onnx import numpy_helper, helper
import numpy as np
from copy import deepcopy
from onnxsim import simplify


def matmul_to_conv2d(node, init_dict):
    weight_name = node.input[1]
    weight_tensor = init_dict[weight_name]
    weight = numpy_helper.to_array(weight_tensor)
    weight = np.expand_dims(weight.transpose(1,0),[2,3])
    weight_tensor = numpy_helper.from_array(weight, name=weight_name)
    init_dict[weight_name] = weight_tensor

def delete_init(model):
    init_len = len(model.graph.initializer)
    for i in range(init_len):
        model.graph.initializer.pop()

def convert_input_nhwc_nchw(model):
    batch_dim = 1
    dim_list = [dim_val.dim_value for dim_val in model.graph.input[0].type.tensor_type.shape.dim]
    dim_list.insert(0, batch_dim)
    dim_list = np.array(dim_list)[[0,3,1,2]]
   
    input_node = onnx.helper.make_tensor_value_info('input.1', \
                                                    onnx.TensorProto.FLOAT, dim_list.tolist())
    model.graph.input.pop()
    model.graph.input.append(input_node)
    
    dim_list = [dim_val.dim_value for dim_val in model.graph.output[0].type.tensor_type.shape.dim]
    dim_list.insert(0, batch_dim)
    dim_list.insert(2, 1)
    dim_list = np.array(dim_list)[[0,3,1,2]]
   
    out_node = onnx.helper.make_tensor_value_info(model.graph.output[0].name, \
                                                  onnx.TensorProto.FLOAT, dim_list.tolist())
    model.graph.output.pop()
    model.graph.output.append(out_node)
    
def reducemax_to_maxpool(node, model):
    node = helper.make_node(op_type="MaxPool", inputs=node.input, \
                            outputs=node.output, name=node.name,  \
                            ceil_mode = 0, kernel_shape = [1,20], \
                            pads = [0,0,0,0], strides=[1,1])
    model.graph.node.append(node)

def convert_tile(node, init_dict):
    arr_name = node.input[1]
    arr = np.array([1,1,1,20],np.int64)
    tensor = numpy_helper.from_array(arr, name=arr_name)
    init_dict[arr_name] = tensor

def simplify_model(model_path):
    model = onnx.load(model_path)
    if model is None:
        print("File %s is not find! "%model_path)
    return simplify(model)

def simplify_pfe_rpn_model(pfe_model_path, rpn_model_path):
    
    model, check = simplify_model(pfe_model_path)
    if not check:
        print("[ERROR]:Simplify %s error!"% pfe_model_path)    
    onnx.save(model, pfe_model_path)

    model, check = simplify_model(rpn_model_path)
    if not check:
        print("[ERROR]:Simplify %s error!"% rpn_model_path)    
    onnx.save(model, rpn_model_path)



if __name__ == "__main__":

    pfe_model_path = "./onnx_model/pfe_waymo_zf_frame_2000.onnx"
    pfe_model_save_path = "./onnx_model/pfe_waymo_zf_frame_2000_sim.onnx"

    rpn_model_path = "./onnx_model/rpn_waymo_zf_frame_2000.onnx"
  
    simplify_pfe_rpn_model(pfe_model_path, rpn_model_path)
    
    # modify pfe model
    model = onnx.load(pfe_model_path)
    init_dict = {}
    for init_node in model.graph.initializer:
        init_dict[init_node.name] = init_node
    
    #delete nodes
    delete_dict = {}
    tmp_trans = []
    for node in model.graph.node:
        if node.op_type in {"Transpose", "Expand", "Squeeze"}:
            delete_dict[node.output[0]] = node
            if node.op_type == 'Transpose':
                tmp_trans.append(copy.deepcopy(node))

    val_len = len(model.graph.value_info)
    for idx in range(val_len):
        model.graph.value_info.pop()

    delete_init(model)

    matmul_weight_name = []
    convert_input_nhwc_nchw(model)
    rm_list = []
    for node in model.graph.node:
        
        # convert MatMul to Conv2D
        if node.op_type == "MatMul":
            node.op_type = "Conv"
            matmul_to_conv2d(node, init_dict)
        
        if node.input[0] in delete_dict.keys():
            node.input[0] = delete_dict[node.input[0]].input[0]
        
        # convert ReduceMax to maxpool
        if node.op_type == "ReduceMax":
            rm_list.append(node)
            reducemax_to_maxpool(node, model)
        if node.op_type == "Tile":
            convert_tile(node, init_dict)
        if node.op_type == "Concat":
            node.attribute[0].i = 1
    for node in model.graph.output:
        if node.name in delete_dict.keys():
            node.name = delete_dict[node.name].input[0]

    for name,tensor in init_dict.items():
        model.graph.initializer.append(tensor)
        
    for keys,node in delete_dict.items():
        model.graph.node.remove(node)

    for i, node in enumerate(rm_list):
        print(i, ' ', node.name)
        model.graph.node.remove(node)

    first_node = None
    for i, node in enumerate(model.graph.node):
        if node.op_type == "Conv":
            weights = node.input[1]
            for data in model.graph.initializer:
                if data.name == weights:
                    if data.dims[0] == 32:
                        first_node = i
                        break
    tmp = copy.deepcopy(model.graph.input[0])
    for _ in range(len(model.graph.input)):
        model.graph.input.pop(0)
    tmp.name = 'input.1'
    assert len(tmp.type.tensor_type.shape.dim) == 3
    tmp_dim = copy.deepcopy(tmp.type.tensor_type.shape.dim[0])
    tmp.type.tensor_type.shape.dim.append(tmp_dim)
    tmp.type.tensor_type.shape.dim[0].dim_value = 1
    tmp.type.tensor_type.shape.dim[1].dim_value = 320000
    tmp.type.tensor_type.shape.dim[2].dim_value = 20
    tmp.type.tensor_type.shape.dim[3].dim_value = 10
    model.graph.input.append(tmp)
    tmp_trans[0].input[0] = model.graph.input[0].name
    tmp_trans[0].output[0] = model.graph.input[0].name + '_trans'
    model.graph.node[first_node].input[0] = tmp_trans[0].output[0]
    datas = []
    for _ in range(first_node):
        for input in model.graph.node[0].input:
            for i, data in enumerate(model.graph.initializer):
                if data.name == input:
                    if i not in datas:
                        datas.append(i)
        model.graph.node.pop(0)
    datas = sorted(datas)
    for i, data in enumerate(datas):
        model.graph.initializer.pop(data - i)

    tmp_trans[0].attribute[0].ints[0] = 0
    tmp_trans[0].attribute[0].ints[1] = 3
    tmp_trans[0].attribute[0].ints[2] = 1
    tmp_trans[0].attribute[0].ints.append(2)
    model.graph.node.insert(0, tmp_trans[0])
    last_node = 0
    for i, node in enumerate(model.graph.node):
        if node.output[0] == model.graph.output[0].name:
            last_node = i
    tmp_trans[1].input[0] = model.graph.output[0].name
    tmp_trans[1].output[0] = '47'
    model.graph.output[0].name = '47'
    tmp_trans[1].attribute[0].ints[0] = 0
    tmp_trans[1].attribute[0].ints[1] = 2
    tmp_trans[1].attribute[0].ints[2] = 1
    tmp_trans[1].attribute[0].ints.append(3)
    model.graph.node.append(tmp_trans[1])
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 32000
    model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 64
    model.graph.output[0].type.tensor_type.shape.dim[3].dim_value = 1

    inputs = [a.name for a in model.graph.input]
    outputs = [a.name for a in model.graph.output]
    for node in model.graph.node:
        for a in node.output:
            if a not in (inputs):
                inputs.append(a)
    for node in model.graph.node:
        for a in node.input:
            if a not in (outputs):
                outputs.append(a)
    delete_node = []
    for i, node in enumerate(model.graph.node):
        for a in node.input:
            if a not in outputs:
                if i not in delete_node:
                    delete_node.append(i)
        for a in node.output:
            if a not in inputs:
                if i not in delete_node:
                    delete_node.append(i)
    delete_node = sorted(delete_node)
    for i, node in enumerate(delete_node):
        model.graph.node.pop(node - i)

    print(inputs)
    print(outputs)
    for i, node in enumerate(model.graph.node):
        print(i, ' :', node.name)

    # model, check = simplify(model)
    onnx.save(model, pfe_model_save_path)
    model = onnx.load(rpn_model_path)
    for i, out in enumerate(model.graph.output):
        name = str(int(out.name) - 69)
        for j, node in enumerate(model.graph.node):
            for k, o in enumerate(node.output):
                if o == out.name:
                    model.graph.node[j].output[k] = name
        model.graph.output[i].name = name
    #
    name = 'input.1'
    for j, node in enumerate(model.graph.node):
        for k, i in enumerate(node.input):
            if i == model.graph.input[0].name:
                model.graph.node[j].input[k] = name
    model.graph.input[0].name = name
    indice = rpn_model_path.rfind('.')
    rpn_model_path = list(rpn_model_path)
    rpn_model_path.insert(indice, '_sim')
    rpn_model_path = ''.join(rpn_model_path)
    onnx.save(model, rpn_model_path)
    print("Done")
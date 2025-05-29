from models import ConvConv
import torch
from onnx2keras import onnx_to_keras
import onnx
import tensorflow as tf

def clean_name(name):
    # Replace illegal characters and strip leading/trailing underscores
    return name.replace('/', '_').replace('\\', '_').strip('_')

def sanitize_onnx_model(model):
    # Rename all nodes (operations)
    for node in model.graph.node:
        node.name = clean_name(node.name)
        node.output[:] = [clean_name(o) for o in node.output]
        node.input[:] = [clean_name(i) for i in node.input]

    # Rename graph inputs and outputs
    for tensor in model.graph.input:
        tensor.name = clean_name(tensor.name)
    for tensor in model.graph.output:
        tensor.name = clean_name(tensor.name)

    # Rename value_info (internal tensors)
    for value_info in model.graph.value_info:
        value_info.name = clean_name(value_info.name)

    # Rename initializer tensors
    for initializer in model.graph.initializer:
        initializer.name = clean_name(initializer.name)

    return model

MODE = 'LOTO'
TYPE = 'CC'
i = 8

model = ConvConv.load_from_checkpoint(f"./chkpts/{MODE}_fold_{i}_{TYPE}.ckpt")

dummy_input = torch.randn(1, 500, 6, 1)  # Adjust dimensions as needed
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11,    input_names=['input'], output_names=['output'])
onnx_model = onnx.load('model.onnx')

onnx_model = sanitize_onnx_model(onnx_model)

# Save the sanitized ONNX model
onnx.save(onnx_model, "model_clean.onnx")


k_model = onnx_to_keras(onnx_model, ['input'])  # Replace 'input_0' with your input node name
tf.keras.models.save_model(k_model, 'model.h5')

print("Finish")
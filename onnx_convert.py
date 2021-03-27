# Some standard imports
import io
import numpy as np
import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
# load best saved checkpoint
best_model = torch.load('model/best_model_0.9.pth',map_location=torch.device('cpu'))
best_model = best_model.module
# Input to the model
# best_model.set_swish(memory_efficient=False)
x = torch.randn(1, 3, 480, 640, requires_grad=True)
torch_out = best_model(x)
# Export the model
torch.onnx.export(best_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "geofpn.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

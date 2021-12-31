import torch 
from litmodel import LitClassification
import tensorflow as tf 
from pytorch2keras.converter import pytorch_to_keras
import os
import numpy as np
from torch.autograd import Variable


model_file_checkpoint = "checkpoint/model.ckpt"


model = LitClassification.load_from_checkpoint(model_file_checkpoint)
model.eval()

#  Export to onnx
input_sample = torch.randn((1, 3, 224, 224))
model.to_onnx("model.onnx", input_sample, export_params=True)
assert os.path.isfile("model.onnx")

# Export to torchscript
torch.jit.save(model.to_torchscript(), "model.pt")
assert os.path.isfile("model.pt")

# Export to tflite
input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
input_var = Variable(torch.FloatTensor(input_np))
k_model = pytorch_to_keras(model.model, input_var, [(3, 224, 224)], 
                     verbose=True,
                     change_ordering=True)
converter = tf.lite.TFLiteConverter.from_keras_model(k_model)
tflite_model = converter.convert()
with open(os.path.join(os.getcwd(), 'model.tflite'), 'wb') as f:
    f.write(tflite_model)
assert os.path.isfile("model.tflite")
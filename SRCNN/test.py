import cv2
import numpy as np
import onnx
import onnxruntime
import torch

onnx_model = onnx.load("srcnn.onnx")
onnx.checker.check_model(onnx_model)

input_img = cv2.imread('face.png').astype(np.float32)

# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)
# input_img = np.array(torch.rand(1, 3, 256, 256, dtype=torch.float32))
# input_img = input_img * 256
ort_session = onnxruntime.InferenceSession("srcnn.onnx")
ort_inputs = {'input': input_img}
ort_output = ort_session.run(['output'], ort_inputs)[0]  # 用于模型推理，其第一个参数为输出张量名的列表，第二个参数为输入值的字典。
# 其中输入值字典的 key 为张量名，value为 numpy 类型的张量值

ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
cv2.imwrite("ort.png", ort_output)

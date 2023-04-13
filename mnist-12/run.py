import numpy as np
import cv2
import onnx
import onnxruntime

image = cv2.imread('img.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (28, 28)).astype(np.float32) / 255
input = np.reshape(gray, (1, 1, 28, 28))

onnx_model = onnx.load("mnist-12.onnx")
graph = onnx_model.graph

ort_session = onnxruntime.InferenceSession("mnist-12.onnx")
ort_inputs = {graph.input[0].name: input}
ort_output = ort_session.run([graph.output[0].name], ort_inputs)

num = np.argmax(ort_output)
print("the number is: " + str(num))

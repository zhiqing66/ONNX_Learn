import cv2
import numpy as np
import torch
import torch.onnx
from torch import nn


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


def init_torch_model():
    torch_model = SuperResolutionNet(upscale_factor=4)

    state_dict = torch.load('srcnn.pth')['state_dict']
    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model


model = init_torch_model()

input_img = cv2.imread('face.png').astype(np.float32)

# HWC to NCHW  N:batch C:channel H:height W:weight
input_img = np.transpose(input_img, [2, 0, 1])  # transpose()函数的作用是调换数组的行列值的索引值 HWC->CHW
input_img = np.expand_dims(input_img, 0)  # 扩展数组的形状,如果axis=0，那么会在最外层加一个[]

# Inference
torch_output = model(torch.from_numpy(input_img)).detach().numpy()

# NCHW to HWC
torch_output = np.squeeze(torch_output, 0)  # 删除第0维
torch_output = np.clip(torch_output, 0, 255)  # 限制最大值和最小值
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)  # int8:0-255

# Show image
cv2.imwrite("face_torch.png", torch_output)  # cv2.imwrite() 只能保存 BGR 3通道图像，或 8 位单通道图像、或 PNG/JPEG/TIFF 16位无符号单通道图像


# with torch.no_grad():
#     torch.onnx.export(
#         model,
#         torch.randn(1, 3, 256, 256),
#         "srcnn.onnx",
#         opset_version=11,
#         input_names=['input'],
#         output_names=['output'])

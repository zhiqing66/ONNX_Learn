from flask import Flask
import cv2
import numpy as np
import json

app = Flask(__name__)


@app.route('/')
def start():
    input_img = cv2.imread('./SRCNN/face.png').astype(np.float32)
    input_img = np.transpose(input_img, [2, 0, 1])
    input_img = np.expand_dims(input_img, 0)
    input = json.dumps(input_img.tolist())
    res = {
        'data': input
    }
    return res


if __name__ == '__main__':
    app.run()

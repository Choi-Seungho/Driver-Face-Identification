from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, MTCNN
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import os
import argparse
from copy import deepcopy
import yaml
import cv2


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def run(opt):

    with open(opt.names, 'r') as f:
        names = yaml.load(f, Loader=yaml.FullLoader)

    detector = MTCNN(device=device, post_process=True)
    classifier = InceptionResnetV1(classify=True, num_classes=len(names))
    classifier.load_state_dict(torch.load(opt.model))
    classifier.eval()

    try:
        source = int(opt.source)
    except:
        source = opt.source

    cap = cv2.VideoCapture(source)
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break

        faces, _ = detector.detect(frame[:, :, [2,1,0]])
        if not isinstance(faces, type(None)):
            for i, face in enumerate(faces):
                x1, y1, x2, y2 = list(map(int, face))
                img = pre_processing(cv2.resize(frame[y1:y2, x1:x2, [2,1,0]], (opt.size, opt.size), interpolation=cv2.INTER_CUBIC))
                result = softmax(classifier(img).detach().cpu().numpy())
                if opt.target == np.argmax(result):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

def pre_processing(image):
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image.astype(np.float32), axis=0)
    image = fixed_image_standardization(torch.tensor(image))
    return image

def softmax(x):
    exp_a = np.exp(x)
    sum_exp = np.sum(exp_a)
    y = exp_a / sum_exp

    return y

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--names", type=str, required=True)
    parser.add_argument("--size", default=256, type=int)
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--target", type=int, default=1)

    return parser.parse_args()

if __name__ == "__main__":
    opt = parse()
    run(opt)
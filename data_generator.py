from facenet_pytorch import MTCNN
import cv2
import os
import sys
import torch
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def run(opt):
    
    
    cls_name = opt.cls_name
    size = opt.size

    save_path = os.path.join(opt.save_path, cls_name)
    if os.path.isdir(save_path) == False:
        os.makedirs(save_path)
    count = len(os.listdir(save_path))

    mtcnn = MTCNN(device=device)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break

        faces, _ = mtcnn.detect(frame[:, :, [2,1,0]])
        if not isinstance(faces, type(None)):
            areas = list(map(lambda x: (x[2]-x[0]) * (x[3]-x[1]), faces))
            for i, face in enumerate(faces):
                x1, y1, x2, y2 = list(map(int, face))
                if i == areas.index(max(areas)):
                    try:
                        filename = os.path.join(save_path, f"{cls_name}_{count}.jpg")
                        count += 1
                        cv2.imwrite(filename, cv2.resize(frame[y1:y2, x1:x2, :], (size, size), interpolation=cv2.INTER_CUBIC))
                    except:
                        pass
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./dataset")
    parser.add_argument("--cls_name", type=str, default="other")
    parser.add_argument("--size", type=int, default=256)

    return parser.parse_args()

if __name__ == "__main__":
    opt = parse()
    run(opt)
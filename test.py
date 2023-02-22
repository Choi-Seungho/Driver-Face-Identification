import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN



if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    detector = MTCNN(device=device, post_process=True)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break

        faces, prob = detector.detect(frame[:, :, [2,1,0]])
        faces = list(faces)
        print(prob)
        if faces:
            for face in faces:
                x1, y1, x2, y2 = list(map(int, face))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))


        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
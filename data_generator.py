from facenet_pytorch import MTCNN
import cv2
import os
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    if os.path.isdir("dataset") == False:
        os.mkdir("dataset")
    
    if len(sys.argv) != 3:
        print("Insufficient arguments")
        sys.exit()
    
    cls_id = sys.argv[1]
    size = int(sys.argv[2])

    path = os.path.join("dataset", cls_id)
    if os.path.isdir(path) == False:
        os.mkdir(path)
    count = len(os.listdir(path))

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
                        filename = os.path.join(path, f"{cls_id}_{count}.jpg")
                        count += 1
                        cv2.imwrite(filename, cv2.resize(frame[y1:y2, x1:x2, :], (size, size), interpolation=cv2.INTER_CUBIC))
                    except:
                        pass
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
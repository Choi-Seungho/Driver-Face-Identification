from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import numpy as np
import os
import argparse
import tqdm
from utils import accuracy

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test(opt):
    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization,
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    loss_fn = torch.nn.CrossEntropyLoss()

    dataset = datasets.ImageFolder(opt.dataset, transform=trans)
    loader = DataLoader(dataset, num_workers=opt.num_workers, batch_size=opt.batch_size)
    classifier = InceptionResnetV1(classify=True, num_classes=len(dataset.class_to_idx)).to(device)
    classifier.load_state_dict(torch.load(opt.model))
    classifier.eval()

    iterator = tqdm.tqdm(loader, desc=f"Loss 0.0 Acc 0.0 F1 0.0 Precision 0.0 Recall 0.0", dynamic_ncols=True)

    loss, acc, f1, precision, recall = .0, .0, .0, .0, .0

    for batch, (x, y) in enumerate(iterator):
        x = x.to(device)
        y = y.to(device)
        y_pred = classifier(x)

        y, y_pred = y.detach().cpu(), y_pred.detach().cpu()

        loss_batch = loss_fn(y_pred, y)
        loss += loss_batch
        acc += accuracy(y_pred, y)

        iterator.set_description(f"Loss {loss / (batch+1) :.3f} Acc {acc / (batch+1) :.3f}")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dataset", type=str, default=os.path.join(ROOT_DIR, "dataset"))

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse()
    test(opt)
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
from copy import deepcopy


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(opt):
    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization,
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    dataset = datasets.ImageFolder(opt.dataset, transform=trans)
    nums = np.arange(len(dataset))
    trains = nums[:int(.8 * len(nums))]
    valids = nums[int(.8 * len(nums)):]

    train_loader = DataLoader(dataset, num_workers=opt.num_workers, batch_size=opt.batch_size, sampler=SubsetRandomSampler(trains))
    val_loader = DataLoader(dataset, num_workers=opt.num_workers, batch_size=opt.batch_size, sampler=SubsetRandomSampler(valids))

    loss_fn = torch.nn.CrossEntropyLoss()

    resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=len(dataset.class_to_idx)
    ).to(device)

    optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, [5, 10])

    print('\n\nInitial')
    print('-' * 10)
    resnet.eval()
    run_epoch(resnet, loss_fn, val_loader, optimizer, scheduler, device)

    best_loss, best_acc = 100., .0
    best_model = None
    best_epoch = 0

    for epoch in range(opt.epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, opt.epochs))
        print('-' * 10)

        resnet.train()
        loss, acc = run_epoch(resnet, loss_fn, train_loader, optimizer, scheduler, device)

        if loss < best_loss:
            best_loss = loss
            best_model = deepcopy(resnet)
            best_epoch = epoch + 1

        best_acc = acc if acc > best_acc else best_acc

        resnet.eval()
        run_epoch(resnet, loss_fn, val_loader, optimizer, scheduler, device)
    
    if os.path.isdir(os.path.join(ROOT_DIR, "runs")) == False:
        os.mkdir(os.path.join(ROOT_DIR, "runs"))
    save_path = os.path.join(os.path.join(ROOT_DIR, "runs"), f'{len(os.listdir(os.path.join(ROOT_DIR, "runs")))}')
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    torch.save(resnet.state_dict(), os.path.join(save_path, f"final_epoch.pt"))
    torch.save(best_model.state_dict(), os.path.join(save_path, f"best_epoch_{best_epoch}.pt"))
    
    
def run_epoch(model, loss_fn, loader, optimizer, scheduler, device):

    def accuracy(logits, y):
        _, preds = torch.max(logits, 1)
        return (preds == y).float().mean()

    state = "Train" if model.training else "Valid"
    loss = 0
    acc = 0

    iterator = tqdm.tqdm(loader, desc=f"{state} (0 / {len(loader)} Steps) loss: 0.0 acc: 0.0", dynamic_ncols=True)

    for batch, (x, y) in enumerate(iterator):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss_batch = loss_fn(y_pred, y)

        if model.training:
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()

        loss_batch = loss_batch.detach().cpu()
        loss += loss_batch
        acc += accuracy(y_pred, y).detach().cpu()

        iterator.set_description(f"{state} ({batch+1} / {len(iterator)} Steps) loss: {loss / (batch + 1):.3f} acc: {acc / (batch + 1):.3f}")

    if model.training and scheduler is not None:
        scheduler.step()

    loss = loss / (batch + 1)
    acc = acc / (batch + 1)

    return loss, acc

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dataset", type=str, default=os.path.join(ROOT_DIR, "dataset"))

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse()
    train(opt)
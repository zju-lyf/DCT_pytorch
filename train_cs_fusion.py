import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from PIL import Image
from torchvision import datasets, models, transforms
from network.xcep_cs_fusion import Img_DCT_BlockFusion
from network.transform import xception_data_transforms
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
#root = './data/'
#root = '/home/liangyf/env/DCTAnalysis/ws/DCTAnalysis/data/FF/df/c23_df/image_dct/'

def default_loader(path):
    return Image.open(path).convert('RGB')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')
        lines = fh.readlines()
        #print (len(lines))
        imgs = []
        for line in lines:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = line.split()
            imgs.append((words[0], words[1], int(words[2])))
        #print (words[0])
        #print(words[1])
        #print(words[2])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __len__(self):
        #print (len(self.imgs))
        return len(self.imgs)

    def __getitem__(self, index):
        fn, fn1, label = self.imgs[index]
        img = self.loader(fn)
        img = img.resize((224, 224))
        dct = self.loader(fn1)
        dct = dct.resize((224, 224))
        if self.transform is not None:
            img = self.transform(img)
            dct = self.transform(dct)
        #img_dct = torch.cat([img,dct],dim = 0)
        return img, dct, label
def main():
    post_function = nn.Softmax(dim=1)
    args = parse.parse_args()
    root = args.root
    name = args.name
    #fusion = args.fusion
    fusion = ('fc')
    continue_train = args.continue_train
    epoches = args.epoches
    batch_size = args.batch_size
    model_name = args.model_name
    model_path = args.model_path
    output_path = os.path.join('./model/c40_NT', name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    torch.backends.cudnn.benchmark=True
    train_dataset = MyDataset(txt=root + 'train.txt', transform=xception_data_transforms['train'])
    val_dataset = MyDataset(txt=root + 'test.txt', transform=xception_data_transforms['test'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    model = Img_DCT_BlockFusion(num_classes=2,fusion=fusion)
    model = nn.DataParallel(model)
    if continue_train:
        model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    iteration = 0
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    for epoch in range(epoches):
        print('Epoch {}/{}'.format(epoch+1, epoches))
        print('-'*10)
        model=model.train()
        for (image, dct ,labels) in train_loader:
            #print(image.shape)
            #print (type(image))
            image = image.cuda()
            dct = dct.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            input = (image,dct)
            outputs = model(input)
            #outputs = torch.mean(torch.stack([image, dct]), dim=0)
            outputs = post_function(outputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            acc = accuracy(outputs.data, labels, topk = (1,))
            train_acc.update(acc[0],image.shape[0])
            train_loss.update(loss.data.item(),image.shape[0])
            iteration += 1
            if not (iteration % 20):
                print(('iteration {}\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'Prec@1 {train_acc.val:.3f} ({train_acc.avg:.3f})\t'.format(
                    iteration, loss=train_loss,
                    train_acc=train_acc)))
        epoch_loss = train_loss.avg
        epoch_acc = train_acc.avg
        print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        model.eval()
        with torch.no_grad():
            for (image, dct, labels) in val_loader:
                image = image.cuda()
                dct = dct.cuda()
                labels = labels.cuda()
                input = (image, dct)
                outputs = model(input)
                outputs = post_function(outputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                acc = accuracy(outputs.data, labels, topk=(1,))
                val_acc.update(acc[0], image.shape[0])
                val_loss.update(loss.data.item(), image.shape[0])
            epoch_loss = val_loss.avg
            epoch_acc = val_acc.avg
            print('epoch Acc: {:.4f}'.format(epoch_acc))
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        scheduler.step()
        if not (epoch % 10):
            torch.save(model.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
    print('Best val Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    #torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))
    torch.save(model.state_dict(), os.path.join(output_path, "best.pkl"))



if __name__ == '__main__':
    parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--name', '-n', type=str, default='xcep_cs_idct')
    parse.add_argument('--root', '-rt' , type=str, default = '/media/Data/dataset_lyf/output/manipulated_sequences/NeuralTextures/c40/2/')
    #parse.add_argument('--fusion', '-f' , type=list, default = 'block1')
    parse.add_argument('--batch_size', '-bz', type=int, default=32)
    parse.add_argument('--epoches', '-e', type=int, default='50')
    parse.add_argument('--model_name', '-mn', type=str, default='xception.pkl')
    parse.add_argument('--continue_train', type=bool, default=False)
    parse.add_argument('--model_path', '-mp', type=str, default='./model/c40_NT/xcep_cs_idct/best.pkl')
    main()

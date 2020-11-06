import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
import pdb
from torchvision import datasets, models, transforms
from network.xception_s import xception_spatial
from network.xception1 import xception_1
from network.transform import xception_data_transforms
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
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
def cut_img(image):
    w, h= image.size
    item_width = int(w/3)
    box_list = []
    for i in range(0,3):
        for j in range(0,3):
            box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
            box_list.append(box)
    image_list = [image.crop(box) for box in box_list]
    return image_list
def main():
    
    args = parse.parse_args()
    name = args.name
    train_path = args.train_path
    val_path = args.val_path
    continue_train = args.continue_train
    epoches = args.epoches
    batch_size = args.batch_size
    model_name = args.model_name
    model_path = args.model_path
    output_path = os.path.join('./model/c40_NT', name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    torch.backends.cudnn.benchmark=True
    #creat train and val dataloader
    train_dataset = torchvision.datasets.ImageFolder(train_path, transform=xception_data_transforms['train'])
    val_dataset = torchvision.datasets.ImageFolder(val_path, transform=xception_data_transforms['val'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    #Creat the model
    model = xception_1(pretrained=True)
    class_num = 2
    channel_in = model.fc.in_features
    model.fc = nn.Linear(channel_in,class_num)
    '''model = xception_spatial(num_classes=2)
    pretrained_dict = xception1.state_dict()
    model_dict = model.state_dict()
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)'''
    model = torch.nn.DataParallel(model)
    '''class_num = 2
    channel_in = model.fc.in_features
    model.fc = nn.Linear(channel_in,class_num)'''

    #model.to(device)
    if continue_train:
        model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.005)
    #optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    #Train the model using multiple GPUs
	#model = nn.DataParallel(model)
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
        model.train()
        #train_loss = 0.0
        #train_corrects = 0.0
        #val_loss = 0.0
        #val_corrects = 0.0
        for (image, labels) in train_loader:
            labels = labels.cuda()
            #pdb.set_trace()
            image=image.cuda()
            outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = accuracy(outputs.data, labels, topk = (1,))
            train_acc.update(acc[0],image.shape[0])
            train_loss.update(loss.data.item(),image.shape[0])
            iteration += 1
            #print (train_acc.val)
            #print (train_loss.avg)
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
            for (image, labels) in val_loader:
                labels = labels.cuda()
                image = image.cuda()
                outputs = model(image)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                acc = accuracy(outputs.data, labels, topk=(1,))
                val_acc.update(acc[0], image.shape[0])
                val_loss.update(loss.data.item(), image.shape[0])
            epoch_loss = val_loss.avg
            epoch_acc = val_acc.avg
            print('epoch val Acc: {:.4f} best_acc: {:.4f}'.format(epoch_acc, best_acc))
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        scheduler.step()
        if not (epoch % 1):
            torch.save(model.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
    print('Best val Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    #torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))
    torch.save(model.state_dict(), os.path.join(output_path, "best.pkl"))



if __name__ == '__main__':
    parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--name', '-n', type=str, default='xception_img_data_SGD_0.0005')
    parse.add_argument('--train_path', '-tp' , type=str, default = '/media/Data/dataset_lyf/output/manipulated_sequences/NeuralTextures/c40/pic/dct/train')
    parse.add_argument('--val_path', '-vp' , type=str, default = '/media/Data/dataset_lyf/output/manipulated_sequences/NeuralTextures/c40/pic/dct/val')
    parse.add_argument('--batch_size', '-bz', type=int, default=32)
    parse.add_argument('--epoches', '-e', type=int, default='50')
    parse.add_argument('--model_name', '-mn', type=str, default='xception.pkl')
    parse.add_argument('--continue_train', type=bool, default=False)
    parse.add_argument('--model_path', '-mp', type=str, default='./model/c40_NT/xception_img_data_SGD_0.0005/best.pkl')
    main()

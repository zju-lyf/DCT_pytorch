from torchvision import transforms
from PIL import Image, ImageDraw
import random
import math
import numpy as np
import torch
transform1 = transforms.Compose([
	transforms.ToTensor(), 
	]
)
class RandomErasing(object):
    def __init__(self, EPSILON = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.EPSILON = EPSILON
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    #img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    #img[1, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    #img[2, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                    #img[:, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(3, h, w))
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[1]
                    # img[0, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(1, h, w))
                return img

        return img
class RandomNoise(object):
    """
    Args:
        probability: The probability that the Random Erasing operation will be performed.
        pt_num: The number of vertices that make up the random polygon.
        sl: Minimum proportion of erased area against input image.
        sh: Maximum proportion of erased area against input image.
        r1: Minimum aspect ratio of erased area.
        mean: Erasing value.
    """
    def __init__(self, probability=0.5, pt_num=20, sl=0.02, sh=0.45, rl=0.35, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean 
        self.pt_num = pt_num
        assert pt_num>=3, 'pt_num less than 3 ...'
        self.sl = sl 
        self.sh = sh 
        self.rl = rl

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img 

        im_h, im_w, im_c = img.shape   # jpeg4py read image is (h, w, c), numpy.ndarray 

        mask = Image.fromarray(np.random.rand(im_h, im_w))
        draw = ImageDraw.Draw(mask)
        #print(type(mask))
        mask = transform1(mask)

        #mask = np.asarray(mask)
        mask_neg = 1-mask
        for cnt in range(3):
            img[:, :, cnt] = img[:, :, cnt] * mask_neg + mask * self.mean[cnt]
        return img 

mesonet_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}
xception_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(0.1),  # 进行随机水平翻转
        transforms.RandomVerticalFlip(0.1),  # 进行随机竖直翻转
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
        RandomNoise(),
        RandomErasing()
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(0.1),  # 进行随机水平翻转
        transforms.RandomVerticalFlip(0.1),  # 进行随机竖直翻转
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
        RandomNoise(),
        RandomErasing()
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(0.1),  # 进行随机水平翻转
        transforms.RandomVerticalFlip(0.1),  # 进行随机竖直翻转
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
        RandomNoise(),
        RandomErasing()
    ]),
}
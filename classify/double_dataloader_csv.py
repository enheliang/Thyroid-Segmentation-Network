from PIL import Image
from torch.utils.data import Dataset
import csv,torch
import numpy as np


class MyDataset(Dataset):
    def __init__(self, csv_dir, root, root1, root2, root3, transform=None, transform1=None, target_transform=None):
        fh = open(csv_dir)
        fh = csv.reader(fh)
        imgs = []
        for line in fh:
            a = ' '.join(line)
            line = a.rstrip()
            words = line.split()              #
            word_dect_1, word_dect_2 = words[0].split('.')
            # b = torch.tensor(float(words[2]),dtype=torch.float32)
            # c = torch.tensor(float(words[3]),dtype=torch.float32)
            imgs.append((str(root) + words[0], int(words[1]), str(root1) + words[0], str(root2) + words[0],
                         str(root3) + words[0]))
            self.imgs = imgs
            self.transform = transform
            self.transform1 = transform1
            self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label, fn_CAM, fenge, fenge2 = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        CAM_img = Image.open(fn_CAM).convert('RGB')
        fenge_img = Image.open(fenge).convert('RGB')
        fenge_img2 = Image.open(fenge2).convert('RGB')
        #ratio = torch.from_numpy(np.array(ratio))
        if self.transform is not None:
            img = self.transform(img)
            CAM_img = self.transform1(CAM_img)
            fenge_img = self.transform1(fenge_img)
            fenge_img2 = self.transform1(fenge_img2)
        return img, label, CAM_img, fenge_img, fenge_img2

    def __len__(self):
        return len(self.imgs)



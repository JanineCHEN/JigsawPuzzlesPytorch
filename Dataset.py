import os
import cv2
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class FoldDataset(Dataset):

    def __init__(self, imgs_dir, idx, permutations, in_channels=1):
        super(FoldDataset, self).__init__()

        self.imgs_dir = imgs_dir
        self.idx = idx
        self.in_channels = in_channels
        self.permutations = permutations # list

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img_path = os.path.join(self.imgs_dir, self.idx[i])

        if self.in_channels > 1:
            img = cv2.imread(img_path)
        else:
            img = cv2.imread(img_path, 0)

        label = random.randint(0, 999)

        img = cv2.resize(img, (225, 225), cv2.INTER_LINEAR)
        # C * H * W
        if self.in_channels == 1:
            img = img[np.newaxis, :]
        else:
            img = img.transpose(2, 0, 1)

        imgclips = []
        for i in range(3):
            for j in range(3):
                clip = img[:, i * 75: (i + 1) * 75, j * 75: (j + 1) * 75]
                randomx = random.randint(0, 10)
                randomy = random.randint(0, 10)
                clip = clip[:, randomx: randomx+64, randomy:randomy+64]

                imgclips.append(clip)

        imgclips = [imgclips[item] for item in self.permutations[label]]
        imgclips = np.array(imgclips)

        return img, torch.from_numpy(imgclips) / 255.0, torch.tensor(label)


if __name__ == '__main__':

    path = '/Volumes/Data/甲状腺（体检科）数据/TUDPE/images'
    permutations = np.load('permutations.npy').tolist()
    dataset = FoldDataset(path, os.listdir(path), permutations, in_channels=1)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True, shuffle=True)
    for batch in dataloader:
        ii, inputs, labels = batch
        plt.figure()
        for i in range(9):
            plt.subplot(331+i)
            img = inputs[0, i, 0, ...].numpy()
            plt.imshow(img, cmap='gray')
        plt.show()

        print(permutations[labels[0]])

        plt.figure()
        for i in range(9):
            plt.subplot(331 + permutations[labels[0]][i])
            img = inputs[0, i, 0, ...].numpy()
            plt.imshow(img, cmap='gray')
        plt.show()

        plt.figure()
        plt.imshow(ii.squeeze().numpy(), cmap='gray')
        plt.show()
        break









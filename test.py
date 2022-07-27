import numpy as np

# import torch

# print(torch.__version__)
from data_factory.data_loader import get_loader_segment

# data = np.load("dataset/SMD/SMD_train.npy")
# print(data[0].shape)
train_loader = get_loader_segment("dataset/PSM", 256, 100, step=1, mode='train', dataset='PSM')
cnt = 0
for index, (input_data, labels) in enumerate(train_loader):
    print(input_data.shape)
    cnt += 1

print(cnt)

print(518 * 256)

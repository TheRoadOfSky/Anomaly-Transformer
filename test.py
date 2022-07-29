import numpy as np

import torch

from torch import Tensor
# print(torch.__version__)

# from data_factory.data_loader import get_loader_segment
#
# data = np.load("dataset/SMD/SMD_train.npy")
# print(data.shape)
# train_loader = get_loader_segment("dataset/PSM", 256, 100, step=1, mode='train', dataset='PSM')
# cnt = 0
# for index, (input_data, labels) in enumerate(train_loader):
#     print(input_data.shape)
    # cnt += 1

# print(cnt)
#
# print(518 * 256)

t1: Tensor = torch.tensor([[[1, 2],
                    [3, 4],
                    [5, 6]],
                   [[5, 6],
                    [7, 8],
                    [9, 10]]])
t2 = t1
concat0 = torch.concat([t1, t2], dim=0)
concat1 = torch.concat([t1, t2], dim=1)
concat2 = torch.concat([t1, t2], dim=2)
print(concat0)
print(concat1)
print(concat2)
print(concat0.shape)
print(concat1.shape)
print(concat2.shape)
# print(t1.shape)
# print(t1)
# permute = t1.permute(0, 2, 1)
# print(permute)
# print(permute.permute(0, 2, 1))
# print(torch.transpose(t1, 1, 2).shape)
# print(torch.transpose(t1, 1, 2))
#
# print(t1)



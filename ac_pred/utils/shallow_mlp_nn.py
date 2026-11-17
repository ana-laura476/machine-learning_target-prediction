from torch import nn
import torch
import torch.nn.functional as F


class MLP_NN(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(256, 256)
        self.linear_2 = nn.Linear(256, 64)
        self.linear_3 = nn.Linear(64, 1)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.bn_1 = nn.BatchNorm1d(256)
        self.bn_2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x1 = self.dropout_1(F.relu(self.bn_1(self.linear_1(x))))
        x2 = self.dropout_2(F.relu(self.bn_2(self.linear_2(x1))))
        xf = self.linear_3(x2)
        return xf


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['X_train'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}
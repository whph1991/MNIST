import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class Net(BaseModel):
    def __init__(self, dropout_prob):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout_prob = dropout_prob

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

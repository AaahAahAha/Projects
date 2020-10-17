import torch
import torch.nn as nn


class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size_one, hidden_size_two, hidden_size_three , num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size_one)
        self.l2 = nn.Linear(hidden_size_one, hidden_size_two)
        self.l3 = nn.Linear(hidden_size_two, hidden_size_three)
        self.l4 = nn.Linear(hidden_size_three, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        # no activation and no softmax at the end
        return out
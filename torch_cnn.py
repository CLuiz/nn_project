import numpy as np

import torch
from torch.autograd import Variable

from data_util import mnist


class ConvNet(torch.nn.Module):
    def __init__(self, output_dim):
        super(ConvNet, self).__init__()

        self.conv = torch.nn.Sequential()
        self.conv.add_module('conv_1', torch.nn.Conv2d(1, 10, kernel_size=5))
        self.conv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module('relu_1', torch.nn.Relu())
        self.conv.add_module('conv_2', torch.nn.Conv2d(10, 20, kernel_size=5))
        self.conv.add_module('dropout_2', torch.nn.Dropout())
        self.conv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module('relu_2', torch.nn.ReLU())

        self.fc = torch.nn.Sequential()
        self.fc.add_module('fc1', torch.nn.Linear(320, 50))
        self.fc.add_module('relu_3', torch.nn.ReLU())
        self.fc.add_module('dropout_3', torch.nn.Dropout())
        self.fc.add_module('fc2', torch.nn.Linear(50, output_dim))

    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(-1, 320)
        return self.fc.forward(x)


def train(model, loss, optimizer, x_val, y_val):
    x = Variable(x_val, require_grad=False)
    y = Variable(v_val, require_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    fx = model.forward(x)
    output = loss.forward(fx, y)

    output.backward()

    optimizer.step

    return output.data[0]


def predict(model, x_val):
    pass


def main():
    pass


if __name__ == '__main__':
    main()

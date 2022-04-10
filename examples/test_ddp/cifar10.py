# -*- coding: utf-8 -*-

# pai -name pytorch -Dscript="file:///tmp/dist_cifar10.py" -Dvolumes="odps://algo_platform_dev/volumes/pytorch/cifar10" -DworkerCount=2;

from __future__ import print_function
import argparse
import torch
import torchvision.transforms as transforms

from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


classes = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def parse_args():
    '''Parsing input arguments.'''
    parser = argparse.ArgumentParser(description="Arguments for training.")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default='./data/',
        help="Data dir for training and evaluation.",
    )
    args = parser.parse_args()
    return args


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(args):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=False,
        transform=transform)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True)

    test_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=False,
        transform=transform)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    net = Net().cuda()  # GPU

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())  # GPU

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0


    with torch.no_grad():
        ####################################################################
        # The results seem pretty good.
        #
        # Let us look at how the network performs on the whole dataset.

        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()  # GPU
            outputs = net(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def main():
    args = parse_args()
    train(args)

if __name__ == '__main__':
    main()

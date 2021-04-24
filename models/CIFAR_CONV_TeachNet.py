import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F


# 如果有GPU，则使用GPU加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 定义初始化神经网络的权重——标准正态分布
def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

# 超参数的设置
num_epochs = 100
batch_size = 100
learning_rate = 0.001

# CIFAR10 Dataset 数据集
train_dataset = torchvision.datasets.CIFAR10(root='./cifar10', #指定数据集的目录
                            train=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                                                           [0.229, 0.224, 0.225])]),download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./cifar10',
                           train=False,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])]),download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True) #shuffle：在每个Epoch中打乱数据

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

#
# 全卷积网络——student
class ConvNet_teach(nn.Module):
    def __init__(self):
        super(ConvNet_teach, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.globalavgpool = nn.AvgPool2d(8, 8)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout10 = nn.Dropout(0.1)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout10(x)
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.avgpool(x)
        x = self.dropout10(x)
        x = self.bn3(F.relu(self.conv5(x)))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.globalavgpool(x)
        x = self.dropout50(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 如果有GPU，则使用GPU加速
model = ConvNet_teach().to(device)
# 交叉熵和随机梯度下降优化器
criterion = nn.CrossEntropyLoss()
# Adam自带变化的学习率，所以就没有用SGD，懒得动态更改学习率了
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将变量添加到GPU里面，如果有加速
        images = images.to(device)
        labels = labels.to(device)
        # 前馈计算Loss
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 反馈计算梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 输出loss
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


# 测试模型
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# # 保存网络中的参数
torch.save(model.state_dict(), 'cifar_conv_model.ckpt')







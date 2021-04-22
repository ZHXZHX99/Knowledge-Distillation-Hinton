import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 如果有GPU，则使用GPU加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数的设置
num_epochs = 200
batch_size = 100
learning_rate = 0.001

# MNIST Dataset 数据集
train_dataset = torchvision.datasets.MNIST(root='./data', #指定数据集的目录
                            train=True, transform=transforms.Compose([
                                               transforms.RandomAffine(degrees=0,translate=(0.071,0.071)), #随机仿射变换中的平移。
                                               transforms.ToTensor(),
                                               # transforms.ToTensor() 将numpy的ndarray或PIL.Image读的图片转换成形状为(C,H, W)的Tensor格式，且/255归一化到[0,1.0]之间
                                           ]), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True) #shuffle：在每个Epoch中打乱数据

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 两个有400个线性修正激活单元的隐藏层的大网络——student
class NeuralNet_stu(nn.Module):
    def __init__(self):
        super(NeuralNet_stu, self).__init__()
        self.fc1 = nn.Linear(784, 400) # 输入层->隐藏层1
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(400, 400) # 隐藏层1->隐藏层2
        self.fc3 = nn.Linear(400, 10) # 隐藏层2->输出层
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

# 如果有GPU，则使用GPU加速
model = NeuralNet_stu().to(device)
# 交叉熵和随机梯度下降优化器
criterion = nn.CrossEntropyLoss()
# Adam自带变化的学习率，所以就没有用SGD，懒得动态更改学习率了
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将变量添加到GPU里面，如果有加速
        images = images.reshape(-1, 28 * 28).to(device)
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
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))




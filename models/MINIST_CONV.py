import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from ResNet import resnet34

# 如果有GPU，则使用GPU加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数的设置
num_epochs = 100
batch_size = 128
learning_rate = 0.001
alpha = 0.95
Temperature = 15

# MNIST Dataset 数据集
train_dataset = torchvision.datasets.MNIST(root='./data', #指定数据集的目录
                            train=True,
                                           transform=transforms.Compose([
                                               transforms.RandomAffine(degrees=0,translate=(0.071,0.071)), #随机仿射变换中的平移。
                                               transforms.ToTensor(),
                                               # transforms.ToTensor() 将numpy的ndarray或PIL.Image读的图片转换成形状为(C,H, W)的Tensor格式，且/255归一化到[0,1.0]之间
                                           ]),
                            download=True)

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

## 卷积网络
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1,64,3,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU())
        self.conv2 = nn.Sequential(
                nn.Conv2d(64,64,3,1,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU())
        self.conv3 = nn.Sequential(
                nn.Conv2d(64,128,3,1,padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU())
        self.conv4 = nn.Sequential(
                nn.Conv2d(128,128,3,1,padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU())
        self.pooling1 = nn.Sequential(nn.MaxPool2d(2,stride=2))
        self.fc = nn.Sequential(nn.Linear(6272,10))
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pooling1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pooling1(x)
        x = x.view(x.size()[0],-1)
        x = self.fc(x)
        return x
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


# 如果有GPU，则使用GPU加速
# 加载教师网络的参数
model_teach = ConvNet()
model_teach.initialize_weights()
# model_teach.load_state_dict((torch.load('model.ckpt')))
model_teach = model_teach.to(device)


# 设置Loss函数的算子——交叉熵和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_teach.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将变量添加到GPU里面，如果有加速
        # images = images.reshape(-1, 28 * 28).to(device)
        images = images.to(device)
        labels = labels.to(device)
        # 大网络的输出结果——软目标
        soft_target = model_teach(images)
        outputs = model_teach(images)
        # 这里将小网络输出与正确label作为loss1
        loss= criterion(outputs, labels)
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
        images = images.to(device) # .reshape(-1, 28 * 28)
        labels = labels.to(device)
        outputs = model_teach(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# # 保存网络中的参数
torch.save(model_teach.state_dict(), 'model_conv.ckpt')
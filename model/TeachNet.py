import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 如果有GPU，则使用GPU加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数的设置
input_size = 784
hidden_size = 1200
num_classes = 10
num_epochs = 200
batch_size = 100
learning_rate = 0.001

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

# 两个有1200个线性修正激活单元的隐藏层的大网络——teacher
class NeuralNet_teach(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet_teach, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) # 输入层->隐藏层1
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size) # 隐藏层1->隐藏层2
        self.fc3 = nn.Linear(hidden_size, num_classes) # 隐藏层2->输出层
        self.dropout_i = nn.Dropout(p=0.2)  # dropout训练
        self.dropout_h = nn.Dropout(p=0.5)  # 隐藏层的dropout训练
    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout_i(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout_h(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


# 如果有GPU，则使用GPU加速
model = NeuralNet_teach(input_size,hidden_size,num_classes).to(device)

# 设置交叉熵和优化器
criterion = nn.CrossEntropyLoss()
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

# # 保存网络中的参数
torch.save(model.state_dict(), 'model.ckpt')

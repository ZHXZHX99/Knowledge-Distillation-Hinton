# Knowledge-Distillation-Hinton
这是对“神经网络在知识蒸馏”一文中MINIST实验的复现。个人理解与思路详见[博客]()
### papers
这个文件夹里面是论文原文，除了“神经网络中的知识蒸馏”一文，MINIST复现实验参考了一篇Dropout经典文献的设置，所以也有这篇“Improving neural networks by preventing co-adaptation of feature detectors.”

### model
这个文件夹里面就是跟模型有关的py文件。
TeachNet.py中是一个1200的教师网络（784-1200-1200-10）。
StuNet.py中是一个400的学生网络（784-400-400-10）。
Distillation.py是采用知识蒸馏的思想来训练学生网络。
实现效果详见博客内容~


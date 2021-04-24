# Knowledge-Distillation-Hinton
这是对“神经网络在知识蒸馏”一文中MINIST实验的复现。个人理解与思路详见[博客](https://blog.csdn.net/qq_42312574/article/details/115679660?spm=1001.2014.3001.5501)
### papers
这个文件夹里面是论文原文，除了“神经网络中的知识蒸馏”一文，MINIST复现实验参考了一篇Dropout经典文献的设置，所以也有这篇“Improving neural networks by preventing co-adaptation of feature detectors.”

### model
这个文件夹里面就是跟模型有关的py文件和模型参数文件。
- MINIST_FC.py中是对教师/学生网络进行单独训练、测试的文件。对于1200的教师网络（784-1200-1200-10），训练完成之后可以保存模型参数，以便在蒸馏的时候得到模型输出——软目标；对于学生网络而言，就是对比应用软目标训练的前后，测试精度的变化。
- MINIST_KD.py中是采用知识蒸馏的思想来训练学生网络的文件。
- MINIST_CONV.py中是将4层卷积网络作为教师网络，训练并保存参数。
- CIFAR_CONV_TeachNet.py是将6层卷积网络作为教师网络训练，使用了Cifar10数据集，并保存参数。
- CIFAR_FC_StuNet.py是将4层全连接网络（（32x32x4-1200-1200-10））作为学生网络训练，使用了Cifar10数据集，并保存参数。
- CIFAR_KD.py是采用知识蒸馏的思想来训练学生网络的文件。
- model.ckpt保存的是MINIST数据集训练的1200个隐藏层的教师网络，500个epoch的训练，精度98.98%。
- model_conv.ckpt保存的是4层卷积网络在MINIST数据集上运行了200个epoch，精度99.54%。
实现效果详见博客内容~


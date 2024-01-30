import torch
import torch.nn as nn  # torch.nn：包含用于构建神经网络的模块和可扩展类的子包。
import torchvision  # torchvision ：一个提供对流行数据集、模型架构和计算机视觉图像转换的访问的软件包
import torch.utils.data as Data  # torch.utils ：工具包，包含数据集和数据加载程序等实用程序类的子包，使数据预处理更容易

torch.manual_seed(1)  # 设置随机种子, 用于复现

# 超参数
EPOCH = 1  # 前向后向传播迭代次数
LR = 0.001  # 学习率 learning rate
BATCH_SIZE = 50  # 批量训练时候一次送入数据的size
DOWNLOAD_MNIST = True

# 下载mnist手写数据集
# 训练集
train_data = torchvision.datasets.MNIST(
    root='./MNIST/',  # 表示数据集的根目录
    train=True,  # 如果为True，则从training.pt创建数据集，否则从test.pt创建数据集
    transform=torchvision.transforms.ToTensor(),  # 接收PIL图片并返回转换后版本图片的转换函数(就图片或者numpy中的数组转换成tensor)
    download=DOWNLOAD_MNIST  # 如果为True，则从internet下载数据集并将其放入根目录。如果已下载，则不会再次下载
)

# 测试集
test_data = torchvision.datasets.MNIST(root='./MNIST/', train=False)  # train设置为False表示获取测试集

# 一个批训练 50个样本, 1 channel通道, 图片尺寸 28x28 size:(50, 1, 28, 28)
# Dataset是数据集的类，主要用于定义数据集。Sampler是采样器的类，用于定义从数据集中选出数据的规则，比如是随机取数据还是按照顺序取等等。Dataloader
# 是数据的加载类，它是对于Dataset和Sampler的进一步包装，即其实Dataset和Sampler会作为参数传递给Dataloader，用于实际读取数据，可以理解为它是这
# 个工作的真正实践者
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True  # shuffle：表示打乱数据顺序
)
#  测试数据预处理；只测试前2000个
test_x = torch.unsqueeze(test_data.data, dim=1).float()[:2000] / 255.0
# squeeze()函数的功能是维度压缩。返回一个tensor（张量），其中 input 中维度大小为1的所有维都已删除。
# 举个例子：如果 input 的形状为 (A×1×B×C×1×D)，那么返回的tensor的形状则为 (A×B×C×D)
# 当给定 dim 时，那么只在给定的维度（dimension）上进行压缩操作，注意给定的维度大小必须是1，否则不能进行压缩。
# shape from (2000, 28, 28) to (2000, 1, 28, 28)
test_y = test_data.targets[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # Net需要继承自nn.Module

        self.conv1 = nn.Sequential(  # nn.Sequential返回的是一个序列容器用于搭建神经网络的模块
            nn.Conv2d(  # 输入的图片 （1，28，28）
                in_channels=1,
                out_channels=16,  # 经过一个卷积层之后 （16,28,28）
                kernel_size=5,
                stride=1,  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                padding=2
            ),
            nn.ReLU(),  # 为激活函数，使用ReLU激活函数有解决梯度消失的作用
            nn.MaxPool2d(kernel_size=2)  # 经过池化层处理，维度为（16,14,14）
            # maxpooling有局部不变性而且可以提取显著特征的同时降低模型的参数，从而降低模型的过拟合
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(  # 输入（16,14,14）
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),  # 输出（32,14,14）
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 输出（32,7,7）
        )

        self.out = nn.Linear(32 * 7 * 7, 10)    # 用于全连接层

    def forward(self, x):
        x = self.conv1(x)  # （batch_size,16,14,14）
        x = self.conv2(x)  # 输出（batch_size,32,7,7）
        x = x.view(x.size(0), -1)  # (batch_size,32*7*7),对tensor进行reshape
        out = self.out(x)  # (batch_size,10)
        return out


cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # 定义优化器
loss_func = nn.CrossEntropyLoss()  # 定义损失函数

for epoch in range(EPOCH):

    for step, (batch_x, batch_y) in enumerate(train_loader):
        pred_y = cnn(batch_x)
        loss = loss_func(pred_y, batch_y)
        optimizer.zero_grad()  # 清空上一层梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新优化器的学习率，一般按照epoch为单位进行更新

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].numpy()  # torch.max(test_out,1)返回的是test_out中每一行最大的数)
            # 返回的形式为torch.return_types.max(
            #           values=tensor([0.7000, 0.9000]),
            #           indices=tensor([2, 2]))
            # 后面的[1]代表获取indices
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

# 打印前十个测试结果和真实结果进行对比
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')

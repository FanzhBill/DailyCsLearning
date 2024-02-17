import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, config):
        super(AlexNet, self).__init__()
        self._config = config
        # 定义卷积层和池化层。
        self.features = nn.Sequential(
            # 这里使用一个11*11的更大窗口来捕捉对象。
            # 同时，步幅为4，以减少输出的高度和宽度。
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数。
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 使用三个连续的卷积层和较小的卷积窗口。
            # 除了最后的卷积层，输出通道的数量进一步增加。
            # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度。
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 全连接层
        self.classifier = nn.Sequential(
            # 这里，全连接层的输出数量是LeNet中的好几倍，使用dropout层来减轻过拟合。
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            # 最后是输出层。由于不同数据集的类别不同，所以用类别数为num_classes。
            nn.Linear(4096, self._config['num_classes'])
        )

    def forward(self, x):
        x = self.features(x)
        # 将连续的维度范围展平为张量。
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def save_model(self):
        # tate_dict其实就是一个字典，包含了模型各层和其参数tensor的对应关系。
        torch.save(self.state_dict(), self._config['model_name'])

    def load_model(self, map_location):
        # map_location参数是用于重定向，比如此前模型的参数是在cpu中的，我们希望将其加载到cuda:0中。
        # 或者我们有多张卡，那么我们就可以将卡1中训练好的模型加载到卡2中，这在数据并行的分布式深度学习中可能会用到。
        state_dict = torch.load(self._config['model_name'], map_location=map_location)
        # 当权重中的key和网络中匹配就加载，不匹配就跳过。如果strict是True，那必须完全匹配，不然就报错。
        self.load_state_dict(state_dict, strict=False)

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Trainer(object):
    # 初始化模型、配置参数、优化器和损失函数
    def __init__(self, model, config):
        self._model = model
        self._config = config
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=config['lr'],
            weight_decay=config['l2_regularization']
        )
        self.loss_func = nn.CrossEntropyLoss()

    # 对单个小批量数据进行训练，包括前向传播、计算损失、反向传播和更新模型参数
    def _train_single_batch(self, images, labels):
        y_predict = self._model(images)

        loss = self.loss_func(y_predict, labels)
        # 先将梯度清零,如果不清零，那么这个梯度就和上一个mini-batch有关
        self._optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 梯度下降等优化器 更新参数
        self._optimizer.step()
        # 将loss的值提取成python的float类型
        loss = loss.item()

        # 计算训练精确度
        # 这里的y_predict是一个多个分类输出，将dim指定为1，即返回每一个分类输出最大的值以及下标
        _, predicted = torch.max(y_predict.data, dim=1)
        return loss, predicted

    def _train_an_epoch(self, train_loader, epoch_id):
        # 训练一个Epoch，即将训练集中的所有样本全部都过一遍
        # 设置模型为训练模式，启用dropout以及batch normalization
        self._model.train()
        total = 0
        correct = 0
        epoch_losses = 0  # 用于存储当前epoch的loss总和
        # 初始化tqdm，指定描述和进度条
        progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch_id}/{self._config["num_epoch"]}]')
        # 从DataLoader中获取小批量的id以及数据
        for batch_id, (images, labels) in enumerate(progress_bar):
            images = Variable(images)
            labels = Variable(labels)
            if self._config['use_cuda'] is True:
                images, labels = images.cuda(), labels.cuda()
            loss, predicted = self._train_single_batch(images, labels)
            # 累加当前epoch的loss
            epoch_losses += loss
            # 计算训练精确度
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
            accuracy_rate = correct / total * 100.0
            # 更新进度条的信息，而不是重新初始化
            progress_bar.set_postfix(
                batch_loss=loss,
                epoch_loss=epoch_losses / (batch_id + 1),
                **{'accuracy rate': f'{accuracy_rate}%'}
            )
        return [epoch_losses / (batch_id + 1), accuracy_rate]
        # print('[Training Epoch: {}] Batch: {}, Loss: {}'.format(epoch_id, batch_id, loss))
        # print('Training Epoch: {} , Loss: {} , accuracy rate: {}%%'.format(epoch_id, loss, correct / total * 100.0))

    def train(self, train_dataset):
        # 是否使用GPU加速
        self.use_cuda()
        # 用于存储每个epoch的loss和accuracy
        epoch_losses = []
        epoch_accuracies = []

        # 绘图
        # 创建一个图形对象和一个坐标轴对象
        fig, ax = plt.subplots()
        ax.set_title('Training Loss and Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss / Accuracy')
        # 绘制损失曲线
        line1, = ax.plot([], [], 'r-', label='Loss')
        # 绘制准确率曲线
        line2, = ax.plot([], [], 'b-', label='Accuracy')
        # 显示图例
        ax.legend()
        # 绘制空图形
        plt.show(block=False)

        for epoch in range(1, self._config['num_epoch'] + 1):
            # print('-' * 20 + ' Epoch {} starts '.format(epoch) + '-' * 20)
            # 构造DataLoader
            data_loader = DataLoader(dataset=train_dataset, batch_size=self._config['batch_size'], shuffle=True)
            # 训练一个轮次
            tmp_list = self._train_an_epoch(data_loader, epoch_id=epoch)

            epoch_losses.append(tmp_list[0])
            epoch_accuracies.append(tmp_list[1].item() / 100)

            # 更新损失曲线
            line1.set_xdata(range(1, len(epoch_losses) + 1))
            line1.set_ydata(epoch_losses)
            # 更新准确率曲线
            line2.set_xdata(range(1, len(epoch_accuracies) + 1))
            line2.set_ydata(epoch_accuracies)

            # 自动调整坐标轴范围
            ax.relim()
            ax.autoscale_view()

            # 显示更新后的图形
            fig.canvas.draw()
            fig.canvas.flush_events()

        # 保存图像，文件名包含epoch的数量
        plt.savefig('training_plot_epoch{}.png'.format( self._config['num_epoch']))

    # 用于将模型和数据迁移到GPU上进行计算，如果CUDA不可用则会抛出异常
    def use_cuda(self):
        if self._config['use_cuda'] is True:
            assert torch.cuda.is_available(), 'CUDA is not available'
            torch.cuda.set_device(self._config['device_id'])
            self._model.cuda()

    # 保存训练好的模型
    def save(self):
        self._model.save_model()

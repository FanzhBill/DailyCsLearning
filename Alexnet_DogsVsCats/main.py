import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from AlexNet import AlexNet
from Trainer import Trainer

# 定义参数配置信息
torch.manual_seed(1)  # 设置随机种子, 用于复现
alexnet_config = \
    {
        'num_epoch': 5,  # 训练轮次数
        'batch_size': 200,  # 每个小批量训练的样本数量
        'lr': 1e-3,  # 学习率
        'l2_regularization': 1e-4,  # L2正则化系数
        'num_classes': 2,  # 分类的类别数目
        'device_id': 0,  # 使用的GPU设备的ID号
        'use_cuda': True,  # 是否使用CUDA加速
        'model_name': 'AlexNet_epoch{}.pt'.format(5)  # 保存模型的文件名
    }

transforms = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.Resize(256),  # 将图片的短边缩放成size的比例，然后长边也跟着缩放，使得缩放后的图片相对于原图的长宽比不变
        transforms.CenterCrop(224),  # 从图片中心开始沿两边裁剪，裁剪后的图片大小为（size*size）
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化
    ])

train_file_path = r"D:\PyCharm\Py_Projects\DeepLearning\CNN\CNN_DogsVsCats\dogs-vs-cats-small\train"
test_file_path = r"D:\PyCharm\Py_Projects\DeepLearning\CNN\CNN_DogsVsCats\dogs-vs-cats-small\test"

if __name__ == "__main__":
    ####################################################################################
    # AlexNet 模型
    ####################################################################################
    train_dataset = datasets.ImageFolder(train_file_path, transforms)
    test_dataset = datasets.ImageFolder(test_file_path, transforms)
    # define AlexNet model
    alexNet = AlexNet(alexnet_config)

    ####################################################################################
    # 模型训练阶段
    ####################################################################################
    # # 实例化模型训练器
    trainer = Trainer(model=alexNet, config=alexnet_config)
    # # 训练
    trainer.train(train_dataset)
    # # 保存模型
    trainer.save()

    ####################################################################################
    # 模型测试阶段
    ####################################################################################
    alexNet.eval()
    alexNet.load_model(map_location=torch.device('cpu'))
    if alexnet_config['use_cuda']:
        alexNet = alexNet.cuda()

    correct = 0
    total = 0
    # 对测试集中的每个样本进行预测，并计算出预测的精度
    for images, labels in DataLoader(test_dataset, alexnet_config['batch_size']):
        images = Variable(images)
        labels = Variable(labels)
        if alexnet_config['use_cuda']:
            images = images.cuda()
            labels = labels.cuda()

        y_pred = alexNet(images)
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        temp = (predicted == labels.data).sum()
        correct += temp
    import matplotlib.image as mpimg

    image = mpimg.imread('training_plot_epoch{}.png'.format(alexnet_config['num_epoch']))
    # 创建一个新的图形对象
    fig, ax = plt.subplots()
    # 显示图像
    ax.imshow(image)
    # 关闭标题和图例
    ax.set_title('')
    ax.legend([])
    # 添加文本信息（测试集准确率）
    plt.figtext(0.5,
                0.1,
                'Accuracy of the model on the test images: %.2f%%' % (100.0 * correct / total),
                color='red',
                fontsize=12,
                ha='center'
                )
    # 关闭坐标轴
    ax.axis('off')
    # 关闭图形交互
    plt.ioff()
    # 保存覆盖原始图片
    plt.savefig('training_plot_epoch{}.png'.format(alexnet_config['num_epoch']), bbox_inches='tight', pad_inches=0)
    # 关闭图形
    plt.close()

    print('Accuracy of the model on the test images: %.2f%%' % (100.0 * correct / total))

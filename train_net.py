'''
总结：源代码作者把简单的Unet网络写得没有必要地复杂
本来Unet网络结构就很简单，把它不断地模块化就很没有必要。

mxnet训练模型、导出模型、加载模型 进行预测（python和C++）
https://blog.csdn.net/u012234115/article/details/80656030
'''
from Unet import *
# from data_pre_processing import *
from data_pre import *
from PIL import Image
import mxnet.gluon as gluon
import mxnet.gluon.loss as gloss
from mxnet import autograd
import time
import os
import cv2
import numpy as np


# from gluoncv.utils.viz import get_color_pallete

tct_train = TCTSegDataset(tct_dir='data/train', is_train=True)
tct_test = TCTSegDataset(tct_dir='data/test', is_train=False)

ctx = mx.gpu(0)
# ctx = mx.cpu()

def train_model(net, train_iter, num_epochs=5, batch_size=2, save_interval=10):
    num_steps = len(tct_train) // batch_size
    # 设置训练参数
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1, 'wd': 0.0005, 'momentum': 0.9,
                                                          'lr_scheduler': mx.lr_scheduler.PolyScheduler(  # 训练计划？
                                                              num_steps * num_epochs, 0.1, 2, 0.0001)})
    soft_loss = gloss.SoftmaxCrossEntropyLoss(axis=1, sparse_label=False)

    for epoch in range(num_epochs):
        t0 = time.time()
        # images:[batch, N, H, W]
        for i, (images, labels) in enumerate(train_iter):
            tmp = time.time()
            images = images.as_in_context(ctx)
            labels = labels.as_in_context(ctx)
            yz = labels[:, 0, :, :]
            labels = nd.stack(yz, 255 - yz, axis=1)
            labels = labels.astype('float32') / 255

            with autograd.record():
                # [batch, 2, H, W], 2 is num_classes
                outputs = net(images.astype('float32'))
                loss = soft_loss(outputs, labels)
                loss.backward()
            # mean the loss using the batch to update the params
            print('Epoch %s/%s, iter:%s, time:%.7f, loss:%s' % (
                epoch, num_epochs, i, time.time() - tmp, loss.mean().asscalar()))
            trainer.step(images.shape[0])  # images.shape[0]是图像的高，images.shape[1]是宽，images.shape[2]是通道数

        cost_time = time.time() - t0
        print('Epoch time:{}'.format(cost_time))  # 一步迭代使用的时间

        if epoch % save_interval == 0 and epoch != num_epochs - 1:  # 每5步保存一次模型
            print('Epoch [{}/{}], Loss:{}'.format(epoch, num_epochs, loss.mean().asscalar()))
            # save the model
            if not os.path.exists('./output'):
                os.mkdir('output')

            prefix_file = './output'
            print('save the model to output...')
            net.hybridize()
            # test the model
            x_test = nd.random.uniform(shape=(1,3,224,224),ctx=mx.gpu())
            net(x_test)  # 这个x_test传入net后就是Unet那里的x
            net.export(path=prefix_file, epoch=epoch)
            print('save finished~~~')

        if epoch == num_epochs - 1:  # 结束时再保存模型
            print('save the model to output...')
            net.hybridize()
            x_test = nd.random.uniform(shape=(1,3,224,224),ctx=mx.gpu())
            net(x_test)
            net.export(path=prefix_file, epoch=epoch)
            print('save finished~~~')


def eval_model(net, test_iter):
    # with open('./data/test/test.txt', 'r') as f:
    #     pic_name = f.read().split()  # 读取test.txt来对预测的结果命名，但是它不是照txt里面的先后顺序

    for i, (images, label) in enumerate(test_iter):
        # shape = [batch_size, N, Height, Width]
        t0 = time.time()
        # print('print images:', images)

        # net and data should be in the same context
        images = images.as_in_context(ctx)
        print('print images:', images)  # <NDArray 1x3x512x512 @gpu(0)>
        save_input_image(images, i)
        outputs = net(images.astype('float32'))
        outputs = nd.softmax(outputs, axis=1)
        # print("print outputs1:", outputs)

        outputs = unnormal(outputs)
        print("print outputs2:", outputs)
        print("print len(outputs):", len(outputs))
        print("print len(outputs[0]):", len(outputs[0]))
        print("print len(outputs[0][0]):", len(outputs[0][0]))
        # print("print len(outputs[0][0][0]):", len(outputs[0][0][0]))
        # print("print len(outputs[0][0][0][0]):", len(outputs[0][0][0][0]))

        # visualize and compute accuracy/mIOU
        visualize(outputs, i)

        print('test iter {}, time:{}'.format(i, time.time() - t0))


def visualize(outputs, i):
    # https://blog.csdn.net/u011321546/article/details/79523115
    # python实现opencv学习五：numpy操作数组输出图片

    # img = np.zeros([512, 512, 3], np.uint8)#zeros:double类零矩阵  创建400*400 3个通道的矩阵图像 参数时classname为uint8
    # while(True):
    #     cv2.imshow("output", outputs)  # 输出一张400*400的白色图片(255 255 255):蓝(B)、绿(G)、红(R)
    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #         break


    # img[:, :, 0] = outputs[:,:,0] # ones([400, 400])是创建一个400*400的全1矩阵，*255即是全255矩阵 并将这个矩阵的值赋给img的第一维
    # img[:, :, 1] = outputs[:,:,1]  # 第二维全是255
    # print("print img for cv2", img)
    outputs = outputs.asnumpy()
    outputs = cv2.merge([outputs, np.zeros([512, 512], dtype=np.uint8)])
    # outputs = cv2.cvtColor(outputs, cv2.COLOR_RGB2GRAY)

    if os.path.exists('./result/test/') is False:
        os.makedirs('./result/test/')
    cv2.imwrite('./result/test/{}_predict.jpg'.format(i), outputs)
    # cv2.imshow("outputs{}.jpg".format(i), outputs)  # 输出一张400*400的白色图片(255 255 255):蓝(B)、绿(G)、红(R)
    # # # cv2.destroyAllWindow()
    # cv2.waitKey(1000)

    # while(True):
    #
    #     cv2.imshow("outputs{}.jpg".format(i), outputs)  # 输出一张400*400的白色图片(255 255 255):蓝(B)、绿(G)、红(R)
    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #         break


def unnormal(im, mean=nd.array([0.5228142, 0.54256412]), std=nd.array([0.20718039, 0.20099298])):
    im = im.as_in_context(ctx)
    mean = mean.as_in_context(ctx)
    std = std.as_in_context(ctx)  # 都用GPU计算和生成
    print('print im[0]:', im[0])
    # im = nd.reshape(im[0], shape=(512,512,2))

    # im = np.swapaxes(im[0], 0, 1)
    # im = np.swapaxes(im, 1, 2)  # 更换轴位置(形状）的另一种写法
    im = im[0].transpose((1, 2, 0))

    im = ((im * std + mean) * 255).astype('uint8')

    return im


def save_input_image(image, i):
    # image = np.swapaxes(image[0], 0, 1)
    # image = np.swapaxes(image, 1, 2)
    image = image[0].transpose((1, 2, 0))
    rgb_mean = nd.array([0.5228142, 0.54256412, 0.5228142])
    rgb_mean = rgb_mean.as_in_context(ctx)
    rgb_std = nd.array([0.20718039, 0.20099298, 0.23153676])
    rgb_std = rgb_std.as_in_context(ctx)

    image = ((image * rgb_std + rgb_mean) * 255).astype('uint8')  # 反标准化，(类似归一化）
    image = image.asnumpy()  # 转换成numpy以便cv2查看
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./result/test/{}.jpg'.format(i), image)




if __name__ == '__main__':
    batch_size = 3
    num_epochs = 50
    save_interval = 10

    print('load training data')
    train_iter = gdata.DataLoader(tct_train, batch_size, shuffle=True, last_batch='rollover', num_workers=2)
    for i, (images, labels) in enumerate(train_iter):
        if i == 0:
            print('type(images):', type(images))
            print('type(labels):', type(labels))
            print('images.shape:', images.shape)
            print('labels.shape:', labels.shape)
        else:
            break

    print('load testing data')
    test_iter = gdata.DataLoader(tct_test, 1, shuffle=True, last_batch='rollover', num_workers=2)
    # test的batch_size=1

    net = UNet(num_class=2)  # num_class=2,二类目标分割

    net.collect_params().initialize(ctx=ctx)

    print('training...')

    train_model(net=net, train_iter=train_iter, num_epochs=num_epochs, batch_size=batch_size,
                save_interval=save_interval)

    eval_model(net=net, test_iter=test_iter)

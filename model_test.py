'''本程序用来测试mxnet写的Unet目标分割模型，模型文件有*.params和*.json，都放在output文件夹里'''

import time
import mxnet as mx
import cv2
import matplotlib.pyplot as plt
from Unet import *
from data_pre import *
from train_net import unnormal, visualize, save_input_image

# mxnet加载模型并进行前向推断
# https://blog.csdn.net/lengxiaohe/article/details/81329152
num_batch = 10
img_height = 800
img_width = 400
batch_size = 1
tct_test = TCTSegDataset(tct_dir='./data/test', is_train=False)

def eval_model(net, test_iter):
    for i, (images, label) in enumerate(test_iter):
        # shape = [batch_size, N, Height, Width]
        t0 = time.time()
        # print('print images:', images)


        # net and data should be in the same context
        images = images.as_in_context(mx.gpu(0))
        save_input_image(images, i)
        # outputs = net(images.astype('float32'))
        # outputs = nd.softmax(outputs, axis=1)
        output = net.predict(images.astype('float32'))
        output = unnormal(output)
        print("print output:", output)
        visualize(output, i)


        # visualize and compute accuracy/mIOU
        # visualize(outputs, labels, images, i)

        print('test iter {}, time:{}'.format(i, time.time() - t0))


if __name__=='__main__':
    print("哈哈哈哈")
    test_iter = gdata.DataLoader(tct_test, batch_size, shuffle=True, last_batch='rollover', num_workers=2)
    sym, arg_params, aux_params = mx.model.load_checkpoint("./output/output", 49)  # load with net name and epoch num
    # 这里output是模型的名字，不要加上模型后面的symbol，10是迭代次数，10-1=9，用来加载十步迭代的9号params的。
    print('print sym:', sym)
    mod = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names=None)  # label can be empty
    mod.bind(for_training=False, data_shapes=[("data", (num_batch, 3, img_height, img_width))],
             label_shapes=mod._label_shapes)  # data shape, 1 x 2 vector for one test data record
    mod.set_params(arg_params, aux_params, allow_missing=True)
    print('finished loading model')
    eval_model(net=mod, test_iter=test_iter)

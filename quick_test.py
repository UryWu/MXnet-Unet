# load model and predicate
import mxnet as mx
import numpy as np
import cv2
import os
from data_pre import *
'''这个程序并不能测试目标分割模型，但是靠着这个程序，我才写出可以用的model_test.py'''
# 可参考：mxnet训练模型、导出模型、加载模型 进行预测（python和C++）
# https://blog.csdn.net/u012234115/article/details/80656030

# 移动端unet人像分割模型--1
# https://blog.csdn.net/xiexiecn/article/details/83029787
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
os.environ["MXNET_BACKWARD_DO_MIRROR"] = "1"

def post_process_mask(label, img_cols, img_rows, n_classes, p=0.5):
    pr = label.reshape(n_classes, img_cols, img_rows).transpose([1,2,0]).argmax(axis=2)
    return (pr*255).asnumpy()

def load_image(img, width, height):
    im = np.zeros((height, width, 3), dtype='uint8')
    im[:, :, :] = 128

    if img.shape[0] >= img.shape[1]:
        scale = img.shape[0] / height
        new_width = int(img.shape[1] / scale)
        diff = (width - new_width) // 2
        img = cv2.resize(img, (new_width, height))

        im[:, diff:diff + new_width, :] = img
    else:
        scale = img.shape[1] / width
        new_height = int(img.shape[0] / scale)
        diff = (height - new_height) // 2

        img = cv2.resize(img, (width, new_height))
        im[diff:diff + new_height, :, :] = img

    im = np.float32(im) / 127.5 - 1

    return [im.transpose((2,0,1))]


if __name__ == '__main__':
    print("哈哈哈哈哈")
    batch_size = 1
    num_batch = 10
    img_height = 800
    img_width = 400
    n_classes = 2
    # tct_test = TCTSegDataset(tct_dir='data/test/image', is_train=False)
    # test_iter = gdata.DataLoader(tct_test, batch_size, shuffle=True, last_batch='rollover', num_workers=2)

    # load model
    sym, arg_params, aux_params = mx.model.load_checkpoint("output", 49)  # load with net name and epoch num
    # 这里output是模型的名字，不要加上模型后面的symbol，10是迭代次数，10-1=9，用来加载十步迭代的9号params的。

    mod = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names=None)  # label can be empty
    mod.bind(for_training=False, data_shapes=[("data", (num_batch, 3, img_height, img_width))],
             label_shapes=mod._label_shapes)  # data shape, 1 x 2 vector for one test data record
    mod.set_params(arg_params, aux_params, allow_missing=True)

    # test images
    testimg = cv2.imread('data/test/image/100.jpg')

    testimg = cv2.resize(testimg, (800, 400), interpolation=cv2.INTER_CUBIC)
    # cv2图像缩放https://www.xuebuyuan.com/1971769.html  # https://www.cnblogs.com/jyxbk/p/7651241.html
    while(True):
        cv2.imshow('test', testimg)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    print('print testimg after cv2.imread:', testimg)
    print('print testimg.shape:', testimg.shape)
    print('type(testimg):', type(testimg))
    # 为什么要用while waitKet break : https://blog.csdn.net/Addmana/article/details/54604298
    # cv2.waitkey()实现正常退出:https://blog.csdn.net/wc996789331/article/details/90414496
    img = load_image(testimg, img_width, img_height)
    mod.predict(mx.io.NDArrayIter(data=[img]))

    outputs = mod.get_outputs()[0]

    while (True):
        cv2.imshow('mask', post_process_mask(outputs[0], img_width, img_height, n_classes))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # cv2.imwrite(filepath, img, flag)
    # print (predict_stress) # you can transfer to numpy array
    # for i in predict_stress:
    #     cv2.imshow(i)

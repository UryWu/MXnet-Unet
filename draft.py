import mxnet.gluon as gluon
import mxnet.gluon.loss as gloss
from Unet import *
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np

def GPU_or_not():
    # 验证GPU版mxnet是否安装成功
    a = mx.nd.ones((2, 3), ctx=mx.gpu(0))  #mx.gpu(0)
    b = a * 2 + 1
    print(b.asnumpy())


def draft1():
    rgb_mean = nd.array([0.5228142, 0.54256412, 0.5228142])
    rgb_std = nd.array([0.20718039, 0.20099298, 0.23153676])
    print(rgb_mean)


def tensor_manulpulate():
    a1 = nd.array([[[1,1,2],[2,3,3]],[[4,3,3],[5,4,4]]])
    b1 = np.array([[[8,8,8],[8,8,8]],[[9,9,9],[9,9,9]]])
    print('a1:', a1)
    print('b1:', b1)
    print('type(b1):', type(b1), ' b1.shape:', b1.shape)
    a1[:][:][0] = b1[:][:][1]
    a1[:][:][1] = b1[:][:][0]
    # for i in range(len(a1)):
    #     for j in range(len(a1[0])):
    #         for k in range(len(a1[0][0])):
    #             a1[i][j][k] = b1[i][j][k]

    print('after change a1:', a1)


# tensor_manulpulate()


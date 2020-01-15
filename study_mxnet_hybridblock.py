print(1 * 2 ** 2 ** 3)
print(1 * 2 ** 2 * 3)

import mxnet as mx
from mxnet.gluon import loss as gloss, nn
import mxnet.gluon as gluno

class LeNet(gluno.nn.HybridBlock):
    def __init__(self, classes=10,feature_size=2, **kwargs):
        super(LeNet,self).__init__(**kwargs)

        with self.name_scope():
            self.conv1 = nn.Conv2D(channels=20, kernel_size=5, activation='relu')
            self.conv2 = nn.Conv2D(channels=50, kernel_size=5, activation='relu')
            self.maxpool = nn.MaxPool2D(pool_size=2, strides=2)
            self.flat = nn.Flatten()
            self.dense1 = nn.Dense(feature_size)
            self.dense2 = nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        print('F: ',F)
        print('X: ',x)

        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.flat(x)
        ft = self.dense1(x)
        output = self.dense2(ft)
        return output

if __name__ == '__main__':
    net = LeNet()
    net.initialize()
    x = mx.nd.random.normal(shape=(1,1, 64, 64))
    net(x)
    net.hybridize()
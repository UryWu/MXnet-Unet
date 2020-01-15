import mxnet as mx
import mxnet.gluon.nn as nn

# 什么是HybridSequential()
# https://mxnet.incubator.apache.org/versions/master/api/python/gluon/gluon.html?highlight=hybridsequential#mxnet.gluon.nn.HybridSequential
def ConvBlock(channels, kernel_size):
    out = nn.HybridSequential()
    # with out.name_scope():
    out.add(
        nn.Conv2D(channels, kernel_size, padding=kernel_size // 2, use_bias=False),
        nn.BatchNorm(),
        nn.Activation('relu')
        # 很不理解这个研究生为什么要这样写，直接像下面这样写不就好吗：
        # nn.Conv2D(channels, kernel_size, padding=kernel_size // 2, use_bias=False, activation='relu')
        # 还有这个ConvBlock类就不应该出现，直接在Unet里放卷积层，这样另外搞一个类真是不必要的麻烦
    )
    return out


def down_block(channels):
    out = nn.HybridSequential()
    # with out.name_scope():
    out.add(
        ConvBlock(channels, 3),
        ConvBlock(channels, 3)
    )
    return out


class up_block(nn.HybridBlock):
    def __init__(self, channels, shrink=True, **kwargs):
        super(up_block, self).__init__(**kwargs)
        # with self.name_scope():
        print('channels is:', channels)

        # 上采样使用反卷积函数
        # self.upsampler = nn.Conv2DTranspose(channels=channels, kernel_size=4, strides=2,
        #                                     padding=1, use_bias=False, groups=channels,
        #                                     weight_initializer=mx.init.Bilinear())

        self.upsampler = nn.Conv2DTranspose(channels=channels, kernel_size=4, strides=2,
                                            padding=1, use_bias=False, groups=channels,
                                            weight_initializer=mx.init.Bilinear())

        '''
        # 
        他的这个上采样用的是反卷积，也就是转置卷积，原先在Keras版本里用的是UpSampling2D来上采样
        # weight_initializer=mx.init.Bilinear()？？？
        
        # 这里有一个关于卷积核数量的报错，参照下面的方法来解决：
        mxnet.base.MXNetError: Error in operator unet0_conv10_deconvolution0: [09:16:13] C:\Jenkins\workspace\mxnet-tag\mxnet\src\operator\nn\deconvolution.cc:131: Check failed: d_nchw[1] % param_.num_group == 0U (511 vs. 0) : input num_filter must divide group size
        从零号卷积开始，第十个卷积层就是第一个反卷积的地方。
        
        sulotion1:参考网址：https://discuss.gluon.ai/t/topic/6077
        conv_1 = Conv(data, num_filter=7, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=“conv_1”) # 224/112
        conv_2_dw = Conv(conv_1, num_group=7, num_filter=14, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name=“conv_2_dw”)
        conv_2_dw的num_group==conv_1的num_filter，conv_2_dw的num_filter==conv_2_dw的num_group的倍数
        所以说，这里本层卷积的channels必须是groups的倍数，下一层卷积的groups必须等于上一层卷积的channels
        但是这个解决方法没用。
        
        sulotion2:出现上面的错误的时候我的mxnet版本为1.5.0,更换了1.4.0后，终于过了这个奇葩的错误，因为我根本就没错。
        但是出现新的问题：File "G:/MXnet-Unet/train_net.py", line 100, in <module>
        net.collect_params().initialize(ctx=ctx)
        mxnet.base.MXNetError: [18:47:58] C:\Jenkins\workspace\mxnet-tag\mxnet\src\ndarray\ndarray.cc:1279: GPU is not enabled
        
        参照：https://discuss.mxnet.io/t/mxneterror-gpu-not-enabled/446
        使用命令：pip install mxnet-cu80==1.0.0
        
        之后出现了找不到mxnet模块的错误，很无语出现这种奇葩的错误。我又试了mxnet-cu80==1.4.0也没用
        看了此网址：https://discuss.gluon.ai/t/topic/8390/14
        才知道这个mxnet-cu80的80是指cuda，本机的cuda是9.0的，于是我卸载了80安装了mxnet-cu90==1.4.0
        终于可以运行了，但是错误回到了param_.num_group == 0U (511 vs. 0) : input num_filter must divide group size
        说明更换为mxnet==1.4.0是毫无意义的行为，错误根本没有解决
        
        刚刚看了这个网站：https://blog.csdn.net/marleylee/article/details/81988365
        安装mxnet-cu90是GPU版本，安装mxnet是CPU版本
        
        '''
        self.upsampler.collect_params().setattr('gred_req', 'null')
        '''grad_req ({'write', 'add', 'null'}, default 'write')
        Specifies how to update gradient to grad arrays.
        'null’ means gradient is not requested for this parameter. gradient arrays will not be allocated.'''
        self.conv1 = ConvBlock(channels, 1)
        self.conv3_0 = ConvBlock(channels, 3)
        if shrink:
            self.conv3_1 = ConvBlock(channels // 2, 3)  # 拼接之后卷积通道减半，见Unet结构图更方便理解
        else:
            self.conv3_1 = ConvBlock(channels, 3)

    # 看样子，x对应特征提取部分的x4、y3、y2、y1，s对应上采样的x3、x2、x1、x0，不知道F是什么东西
    def hybrid_forward(self, F, x, s):
        x = self.upsampler(x)  # 先反卷积放大图像，长宽X2
        x = self.conv1(x)  # 这个1X1卷积是用来规范输出的吗？
        x = F.relu(x)

        x = F.Crop(*[x, s], center_crop=True)
        x = F.concat(s, x, dim=1)  # 拼接池化与池化之前的图像，跟keras的concatenate([conv3,up7], axis = 3)类似
        # x = s + x
        x = self.conv3_0(x)  # 拼接之后提取特征，这个conv3_0步长为1，padding为1，卷积核大小为3，(pic_size-3+2*1)+1==pic_size，原图片大小没有变
        x = self.conv3_1(x)  # conv3_1的作用就是：拼接之后卷积通道减半，参见Unet结构图更方便理解
        # conv3_0和conv3_1都是padding=1，没有缩小图像的大小，而Unet是把图像卷积-2了的，而且Unet是先conv3_1后再conv3_0的。
        return x


class UNet(nn.HybridBlock):
    def __init__(self, first_channels=64, num_class=3, **kwargs):
        super(UNet, self).__init__(**kwargs)
        with self.name_scope():
            self.d0 = down_block(first_channels)

            self.d1 = nn.HybridSequential()
            self.d1.add(nn.MaxPool2D(2, 2, ceil_mode=True), down_block(first_channels * 2))
            # 池化为1/2，步长为2.
            self.d2 = nn.HybridSequential()
            self.d2.add(nn.MaxPool2D(2, 2, ceil_mode=True), down_block(first_channels * 2 ** 2))

            self.d3 = nn.HybridSequential()
            self.d3.add(nn.MaxPool2D(2, 2, ceil_mode=True), down_block(first_channels * 2 ** 3))

            self.d4 = nn.HybridSequential()
            self.d4.add(nn.MaxPool2D(2, 2, ceil_mode=True), down_block(first_channels * 2 ** 4))

            self.u3 = up_block(first_channels * 2 ** 3, shrink=True)  # up_block结束时完成了卷积核的减半，所以输出通道为512没有错
            self.u2 = up_block(first_channels * 2 ** 2, shrink=True)
            self.u1 = up_block(first_channels * 2, shrink=True)
            self.u0 = up_block(first_channels, shrink=False)

            # classes = 3-->background, neural, cyto
            self.conv = nn.Conv2D(num_class, 1)

    def hybrid_forward(self, F, x):  # 本函数就是把每层网络连接起来
        print('F：', F)
        x0 = self.d0(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        # print('x4:', x4)
        y3 = self.u3(x4, x3)
        y2 = self.u2(y3, x2)
        y1 = self.u1(y2, x1)
        y0 = self.u0(y1, x0)

        out = self.conv(y0)

        return out

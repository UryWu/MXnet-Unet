#### 项目目标：

把图像中的人去掉，用背景填充。

#### 环境要求：

ubuntu 16.04LTS：

python3.6

mxnet-cu90

numpy

matplotlib

opencv-python

#### 使用：

fill_one_pic_improve_click_area.py 是应用演示

model_test.py 是测试之前生成的模型

train_net.py 是训练模型

trans_gray.py 图片变灰色，那个蓝绿色的Unet目标分割的图转换成黑白二值图

其它的py文件有很多都是不能用的，是我学习别人的代码拷进来的。

#### 应用背景：

​		景区安装了很多摄像头，然后用户自拍的效果不好，要用远处高处的摄像头，这个抠图就是为了提升用户的体验，把除自己之外的其它人物用背景填充掉。但是不能用仅仅用提前拍好的一张背景图来填充，因为这张静态的图是不会根据景区时间、季节的变化而变化。所以要让摄像头在用户拍照前提前拍摄视频，然后从视频中取那么十张图片，把人像都扣掉，平均拼成一张尽量完整的背景图。然后用户拍照的时候选定保留的区域，其它部分用这张背景图填充。

#### 演示：

​		由于没有继续开发，我没有把算法和选取特定区域的应用程序连起来。只能做一个简陋的演示。十张图片是用实验室的2080Ti用Unet网络抠走了人像，抠好后通过fill_one_pic_improve_click_area.py这个程序来平均十张照片的像素生成背景来进行应用演示。

运行fill_one_pic_improve_click_area.py后，在出现的图中拖动鼠标，截取某人，按'q'，未被截取部分的人像消失。

![img](https://github.com/UryWu/MXnet-Unet/blob/master/%E5%BA%94%E7%94%A8%E6%BC%94%E7%A4%BA.gif)

#### 其它：

训练模型基本上要用2080Ti来训练，显存占用很大，用6GB显存以下的显卡训练不了。测试模型虽然只占用4.7GB显存左右，但是我用显存总量为6GB的1060也不能测试。



我的Debug笔记：

https://blog.csdn.net/qq_25799253/article/details/97567920# MXnet-Unet

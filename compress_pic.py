import cv2
import time
from PIL import Image
import os
from glob import glob

fpath = r"F:\Projects\MXnet-Unet\result\test_ending"  #有四个目录 "./data3/validation/1"  "./data3/train/0"  "./data3/train/1"
size = (4096, 2160)  # 定义要调整成为的尺寸(宽度，高度)（PIL会自动根据原始图片的长宽比来缩放适应设置的尺寸，CV压缩则不会）


def plt_compress_pic(fpath, size):
    time1 = time.time()
    # glob.glob()用来进行模糊查询，增加参数recursive=True后可以使用**/来匹配所有子目录
    files = glob(fpath + "**/*.jpg", recursive=True) + glob(fpath + "**/*.png", recursive=True)  # 如果输入的图是jpg,写jpg
    total = len(files)  # 总文件数
    cur = 1  # 当前文件序号
    print("共有" + str(total) + "个文件，开始处理")
    print("-----------------------------------")
    for infile in files:
        try:
            # f, ext = os.path.splitext(infile) # 分离文件名和后缀
            print("进度:" + str(cur) + "/" + str(total) + "   " + infile)
            img = Image.open(infile)  # 打开图片文件
            if img.width > 100:
                img.thumbnail(size, Image.ANTIALIAS)  # 使用抗锯齿模式生成缩略图（压缩图片）
                img.save(infile, "jpeg")  # 保存成与原文件名一致的文件，会自动覆盖源文件，如果是jpg图写jpeg
            else:
                print(infile + "宽度小于1200px，无需处理，已忽略")
            cur = cur + 1

        except OSError:
            print(infile + "文件错误，忽略")
    time2 = time.time()
    print(u'总共耗时：' + str(time2 - time1) + 's')


def cv_compress_pic(fpath, size):
    time1 = time.time()
    files = glob(fpath + "**/*.jpg", recursive=True) + glob(fpath + "**/*.png", recursive=True)  # 如果输入的图是jpg,写jpg
    total = len(files)  # 总文件数
    cur = 1  # 当前文件序号
    print("共有" + str(total) + "个文件，开始处理")
    print("-----------------------------------")
    for infile in files:
        try:
            # f, ext = os.path.splitext(infile) # 分离文件名和后缀
            print("进度:" + str(cur) + "/" + str(total) + "   " + infile)
            img = cv2.imread(infile)  # 打开图片文件
            res = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            # cv2.imshow('image', image)
            # cv2.imshow('resize', res)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite(infile, res)
            cur = cur + 1
        except OSError:
            print(infile + "文件错误，忽略")

    time2 = time.time()
    print(u'总共耗时：' + str(time2 - time1) + 's')


def enlarge_pic(fpath, size):
    time1 = time.time()
    files = glob(fpath + "**/*.jpg", recursive=True) + glob(fpath + "**/*.png", recursive=True)  # 如果输入的图是jpg,写jpg
    total = len(files)  # 总文件数
    cur = 1  # 当前文件序号
    print("共有" + str(total) + "个文件，开始处理")
    print("-----------------------------------")
    for infile in files:
        try:
            # f, ext = os.path.splitext(infile) # 分离文件名和后缀
            print("进度:" + str(cur) + "/" + str(total) + "   " + infile)
            img = cv2.imread(infile)  # 打开图片文件
            res = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)  # interpolation=cv2.INTER_LANCZOS4使用lanczos插值
            # cv2.imshow('image', image)
            # cv2.imshow('resize', res)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite(infile, res)
            cur = cur + 1
        except OSError:
            print(infile + "文件错误，忽略")

    time2 = time.time()
    print(u'总共耗时：' + str(time2 - time1) + 's')


# cv_compress_pic(fpath, size)
enlarge_pic(fpath, size)
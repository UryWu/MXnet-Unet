import cv2, os

target_file_path = './result/test/'


# 将彩色图片转换成灰色图片，不进行备份，直接覆盖
def rgb_to_gray(target_file_path):
    for i in os.listdir(target_file_path):
        print('{0}\\{1}'.format(target_file_path, i))
        img = cv2.imread('{0}\\{1}'.format(target_file_path, i), cv2.IMREAD_GRAYSCALE)
        while(True):
            cv2.imshow('picture', img)
            if cv2.waitKey(1000):
                cv2.destroyAllWindows()
            break
        cv2.imwrite('{0}\\{1}'.format(target_file_path, i), img)


# 将彩色图片转换成黑白二值图片，不进行备份，直接覆盖
def to_binary(target_file_path):
    for i in os.listdir(target_file_path):
        # print('{0}\\{1}'.format(target_file_path, i))
        # gray = cv2.imread('{0}\\{1}'.format(target_file_path, i), cv2.IMREAD_GRAYSCALE)
        gray = cv2.imread('{0}\\{1}'.format(target_file_path, i))
        ret, im_fixed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)  # 0,255将图片灰色部分变白
        # thresh：阈值, maxval：当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值,255是极端强化该颜色，0是弱化该颜色
        # ret就是自己输入的参数thresh：阈值，可以输出看一下。
        cv2.imwrite('{0}\\{1}'.format(target_file_path, i), im_fixed)


# 将灰色图片转换成伪彩色图片，好像用不了
def gray_to_rgb(target_file_path):
    for i in os.listdir(target_file_path):
        # print('{0}\\{1}'.format(target_file_path, i))
        img = cv2.imread('{0}\\{1}'.format(target_file_path, i), cv2.COLOR_GRAY2RGB)
        cv2.imwrite('{0}\\{1}'.format(target_file_path, i), img)


def trans_for_unet():  # 把那个蓝绿的Unet目标分割的图转换成黑白二值图
    for k in os.listdir(target_file_path):
        image = cv2.imread('{0}\\{1}'.format(target_file_path, k))
        for i in range(len(image[0])):
            for j in range(len(image[1])):
                if image[i][j][0]>125:  # 只要蓝色通道的值大于140就是让此像素为白色
                    image[i][j][:] = 255
                else:
                    image[i][j][:] = 0

        # while(True):
        #     cv2.imshow('image', image)
        #     if cv2.waitKey(10) & 0xFF == ord('q'):
        #         break
        cv2.imwrite('{0}\\{1}'.format(target_file_path, k), image)
# rgb_to_gray(target_file_path)
# to_binary(target_file_path)
trans_for_unet()

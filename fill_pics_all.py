# 采用该目录下所有的图片来填充，不需要给定读取图片的数量
import cv2, os
import numpy as np
import time
target_file_path = './result/test/'
num_fill_pic = 8  # 用来填充的图片序号0-?


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass


# 读取目录下所有图片，纯数字名字的图片放入images，带有predict字样的图片放入masks
def read_pics():
    images = np.zeros([1, 512, 512, 3], dtype=np.uint8)  # 加上这个dtype=np.uint8太重要了，否则图片会显示很模糊，或是无法显示
    masks = np.zeros([1, 512, 512, 3], dtype=np.uint8)
    pics = os.listdir(target_file_path)

    image_names = {}

    for k in pics:
        if is_number(k.split('.')[0]):
            image_names[int(k.split('.')[0])] = k  # 为所有纯数字的原图片创建一个字典。

    image_names_sorted = {}
    for i in sorted(image_names):  # 将字典按key排序
        image_names_sorted[i] = image_names[i]
    print(image_names_sorted)

    for k in image_names_sorted:  # 读取文件，将原图放入images，目标分割图放入masks
        # print(k)
        img = cv2.imread('{0}\\{1}'.format(target_file_path, image_names_sorted[k].replace('.', '_predict.')))
        img = img[np.newaxis, :]
        masks = np.vstack((masks, img))

        img = cv2.imread('{0}\\{1}'.format(target_file_path, image_names_sorted[k]))
        img = img[np.newaxis, :]
        images = np.vstack((images, img))

    masks = masks[1:]
    images = images[1:]
    return images,masks
    # for k in images:
    #     cv2.imshow('image', k)
    #     while(True):
    #         if cv2.waitKey(1000):
    #             cv2.destroyAllWindows()
    #         break
    # 将像素值写入文件
    # with open('./1.txt', 'w') as f:
    #     for i in range(len(images[0])):
    #         for j in range(len(images[0][0])):
    #             f.write('{}'.format(images[1][i][j]))
    #         f.write('\n')


# 填入多张图片的平均像素值
def fill_pics(images, masks):
    for num_fill_pic in range(10):  # 对10张图片都填充
        for i in range(len(masks[0])):  # 多张图片的平均像素填充mask
            for j in range(len(masks[0][0])):
                # print('images[1][i][j][:] :', images[1][i][j][:])
                if all(masks[num_fill_pic][i][j][:] >= [240, 240, 240]):  # all用来比较numpy数组里的每个值是否符合，如果当前像素为白色，即是空白处
                    a = np.empty(shape=[1, 3], dtype=np.uint8)

                    for k in range(len(masks)):
                        if all(masks[k][i][j][:] < [240, 240, 240]):  # 如果当前像素处于不被分割出来的区域，也就是mask里的黑区域，就放入a中求平均像素，之后覆盖到mask中
                            a = np.vstack((a, images[k][i][j][:]))  # 采用np.vstack可以实现动态增加numpy的功能

                    a = a[1:]
                    a = [np.mean(a[:, 0]).astype(np.uint8), np.mean(a[:, 1]).astype(np.uint8),np.mean(a[:, 2]).astype(np.uint8)]  # 求多张图片的平均像素
                    # input('breakpoint')
                    masks[num_fill_pic][i][j][:] = a

        # images[1,:,:,:] = np.ones([512,512,3], dtype=np.uint8)*255  # 将图片设置为白色

        # 展示填充后的masks
        cv2.imwrite('{0}\\{1}'.format(target_file_path, '{0}_filled_mask.jpg'.format(num_fill_pic)), masks[num_fill_pic])
        # cv2.imshow('fill in mask{0}'.format(num_fill_pic), masks[num_fill_pic])
        # while(True):
        #     if cv2.waitKey(2000):
        #         cv2.destroyAllWindows()
        #     break

        # 展示填充原图之后
        for i in range(len(masks[0])):
            for j in range(len(masks[0][0])):
                if any(masks[num_fill_pic][i][j][:] > [20, 20, 20]):  # all用来比较numpy数组里的每个值是否符合
                    images[num_fill_pic][i][j][:] = masks[num_fill_pic][i][j][:]
                    # 如果该像素不为黑，就覆盖到原图，就是填充后的mask覆盖到原图，我没有用>[0,0,0]来判断是否为黑，因为稍微调高一点效果好一点


        cv2.imwrite('{0}\\{1}'.format(target_file_path, '{0}_filled_image.jpg'.format(num_fill_pic)), images[num_fill_pic])
        # cv2.imshow('fill in image{0}'.format(num_fill_pic), images[num_fill_pic])
        # while(True):
        #     if cv2.waitKey(2000):
        #         cv2.destroyAllWindows()
        #     break


if __name__ == '__main__':
    startTime = time.time()  # 开始时刻
    images, masks = read_pics()  # 读取图片
    fill_pics(images, masks)  # 填充图片
    endTime = time.time()  # 结束时刻
    print('time using:{0:.2f}s'.format(endTime-startTime))

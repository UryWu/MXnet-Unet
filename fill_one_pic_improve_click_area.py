'''
功能：本程序能将target_file_path下的所有的纯数字的图片，通过确定图片号码num_fill_pic来除去人像，但target_file_path目录下
每张纯数字图片必须存在一张纯数字_predict目标分割图片与之对应，如原图为0.jpg，必须存在0_predict.jpg本程序才能正常运行。
'''
import cv2, os
import numpy as np
import time
target_file_path = './result/test/'
num_fill_pic = 10  # 用来填充的图片序号0-?
image_width = 512
image_height = 512
g_rect = [0, 0, 0, 0]


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


def on_mouse(event, x, y, flags, param):  # 鼠标点击交互函数
    global img1, point1, point2, g_rect
    img2 = img1.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
        print("1-EVENT_LBUTTONDOWN")
        point1 = (x, y)
        print("point1:", point1)
        cv2.circle(img2, point1, 8, (0, 255, 0), thickness=2)  # cv2.circle(img2, point1, 8(这是圈的半径), (0, 255, 0), thickness=2(这是圈的粗细))
        cv2.imshow('image', img2)

    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳，画框
        print("2-EVENT_FLAG_LBUTTON")
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), thickness=2)
        cv2.imshow('image', img2)

    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放，显示
        print("3-EVENT_LBUTTONUP")
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), thickness=2)
        cv2.imshow('image', img2)
        print("point1:{0} point2:{1}".format(point1, point2))
        if point1 != point2:
            min_x = min(point1[0], point2[0])
            min_y = min(point1[1], point2[1])
            width = abs(point1[0] - point2[0])
            height = abs(point1[1] - point2[1])
            g_rect = [min_x, min_y, width, height]
            cut_img1 = img1[min_y:min_y + height, min_x:min_x + width]
            # cv2.imshow('ROI', cut_img1)  # 展示截取的图片


# 读取目录下所有图片，纯数字名字的图片放入images，带有predict字样的图片放入masks
def read_pics():
    images = np.zeros([1, image_height, image_width, 3], dtype=np.uint8)  # 加上这个dtype=np.uint8太重要了，否则图片会显示很模糊，或是无法显示
    masks = np.zeros([1, image_height, image_width, 3], dtype=np.uint8)
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
    # for k in images:
    #     cv2.imshow('image', k)
    #     while(True):
    #         if cv2.waitKey(1000):
    #             cv2.destroyAllWindows()
    #         break

    return images,masks

    # 将像素值写入文件
    # with open('./1.txt', 'w') as f:
    #     for i in range(len(images[0])):
    #         for j in range(len(images[0][0])):
    #             f.write('{}'.format(images[1][i][j]))
    #         f.write('\n')


# 填入多张图片的平均像素值
def fill_pics(images, masks):

    global img1
    img1 = images[num_fill_pic].copy()  # copy给保存截图交互用
    for i in range(len(masks[0])):  # 多张图片的平均像素填充mask
        for j in range(len(masks[0][0])):
            # print('images[1][i][j][:] :', images[1][i][j][:])
            if all(masks[num_fill_pic][i][j][:] >= [240, 240, 240]):  # all用来比较numpy数组里的每个值是否符合，如果当前mask的像素为白色，即是mask空白处
                a = np.empty(shape=[1, 3], dtype=np.uint8)

                for k in range(len(masks)):
                    if all(masks[k][i][j][:] < [240, 240, 240]):  # 如果当前像素处于不被分割出来的区域，也就是mask里的黑区域，就放入a中求平均像素，之后覆盖到mask中
                        a = np.vstack((a, images[k][i][j][:]))  # 采用np.vstack可以实现动态增加numpy的功能

                a = a[1:]
                a = [np.mean(a[:, 0]).astype(np.uint8), np.mean(a[:, 1]).astype(np.uint8),np.mean(a[:, 2]).astype(np.uint8)]  # 求多张图片的平均像素
                # input('breakpoint')
                images[num_fill_pic][i][j][:] = a

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_mouse)
    cv2.imshow("image", img1)
    while(True):
        try:
            if cv2.waitKey(100) & 0xff == ord('q'):
                break
        except Exception:
            cv2.destroyWindow("image")
            break
    # g_rect = [min_x, min_y, width, height]
    global g_rect
    print('g_rect', g_rect)
    min_x, min_y, width, height = g_rect[0], g_rect[1], g_rect[2], g_rect[3]
    images[num_fill_pic, min_y:min_y + height, min_x:min_x + width] = img1[min_y:min_y + height, min_x:min_x + width]

    # images[1,:,:,:] = np.ones([512,512,3], dtype=np.uint8)*255  # 将图片设置为白色

    cv2.imwrite('{0}\\{1}'.format(target_file_path, '{0}_filled_image.jpg'.format(num_fill_pic)), images[num_fill_pic])
    cv2.imshow('fill in image{0}'.format(num_fill_pic), images[num_fill_pic])
    while(True):
        if cv2.waitKey(3500):
            cv2.destroyAllWindows()
        break


if __name__ == '__main__':
    startTime = time.time()  # 开始时刻
    images, masks = read_pics()  # 读取图片
    fill_pics(images, masks)  # 填充图片
    endTime = time.time()  # 结束时刻
    print('time using:{0:.2f}s'.format(endTime-startTime))
    # 10张图片平均像素来填10张图片，用时19.17s

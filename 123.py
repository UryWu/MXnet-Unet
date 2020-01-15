# coding: utf-8
import cv2
import numpy as np

# opencv读取图片
def cv2_read():
    img1 = cv2.imread('./1.png')
    print(type(img1))
    print(img1.shape)
    while(True):
        cv2.imshow('pic1', img1)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


# 在图片上点击后显示点和坐标的一种方法（使用opencv python）
# get_pic_x_y()来源：https://blog.csdn.net/huzhenwei/article/details/82900715
# on_mouse()来源：https://blog.csdn.net/guyuealian/article/details/88013421
img1 = cv2.imread("./1.png")
def on_mouse(event, x, y, flags, param):
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
            cv2.imshow('ROI', cut_img1)
def get_pic_x_y():  # 运行这个函数，上面那个被这个调用
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_mouse)

    while (True):
        try:
            cv2.imshow("image", img1)
            if cv2.waitKey(100) & 0xff == ord('q'):
                break
        except Exception:
            cv2.destroyWindow("image")
            break
    # cv2.waitKey(100)
    # cv2.destroyAllWindow()


get_pic_x_y()


def get_pic_x_y():  # 运行这个函数，上面那个被这个调用
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_mouse)
    cv2.imshow("image", img1)
    while (True):
        try:
            cv2.waitKey(100)
        except Exception:
            cv2.destroyWindow("image")
            break
    cv2.waitKey(100)
    cv2.destroyAllWindow()
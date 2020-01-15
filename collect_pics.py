import os
import random
import shutil
import re

GT_from_PATH = r"F:\Data\途拍人像抠图\temp"  # 你的那些存有目标分割标注图片的文件夹的父目录
GT_to_PATH = r"F:\Data\途拍人像抠图\label"  # 寻找目标分割标注图片后放入的目录


def copy_file(from_dir, to_dir, Name_list):
    if not os.path.isdir(to_dir):
        os.mkdir(to_dir)
    # 1
    # name_list = os.listdir(from_dir)

    # # 2
    # sample = random.sample(pathDir, 2)
    # print(sample)

    # 3
    for name in Name_list:
        try:
            # print(name)
            if not os.path.isfile(os.path.join(from_dir, name)):
                print("{} is not existed".format(os.path.join(from_dir, name)))
            shutil.copy(os.path.join(from_dir, name), os.path.join(to_dir, name))
            # print("{} has copied to {}".format(os.path.join(from_dir, name), os.path.join(to_dir, name)))
        except:
            # print("failed to move {}".format(from_dir + name))
            pass
        # shutil.copyfile(fileDir+name, tarDir+name)
    print("{} has copied to {}".format(from_dir, to_dir))


if __name__ == '__main__':
    filepath_list = os.listdir(GT_from_PATH)
    # print(name_list)
    for i, file_path in enumerate(filepath_list):
        gt_path = "{}/{}.png".format(os.path.join(GT_from_PATH, filepath_list[i]), file_path[:-5])
        print("copy {} to ...".format(gt_path))
        gt_name = ["{}.png".format(file_path[:-5])]
        gt_file_path = os.path.join(GT_from_PATH, file_path)
        copy_file(gt_file_path, GT_to_PATH, gt_name)

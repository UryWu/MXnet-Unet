import os
test_or_train = 'train'  # 本程序生成图片文件的txt文档，此处只可填'test'或者'train'


file_path = './data/{0}/image'.format(test_or_train)
file_name_list = []
for i in os.listdir(file_path):
    file_name_list.append(i.split("/")[-1].split('.')[0])
with open("data/{0}/{1}val.txt".format(test_or_train, test_or_train), 'w') as file:
    for i in file_name_list:
        file.write(i+"\n")


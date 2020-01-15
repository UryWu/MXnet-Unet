# from mean_std import get_mean_std
from mxnet import image
import mxnet.gluon.data as gdata
import mxnet.ndarray as nd
import cv2
# just resize to the same size
# mean_, std_ = get_mean_std()


def read_person_data(root='data', is_train=True):
    txt_fname = '%s/%s' % (root, 'trainval.txt' if is_train else 'test.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()

    # images = images[:10]

    features, labels = [None] * len(images), [None] * len(images)

    for i, fname in enumerate(images):
        # print(fname)  # 输出读取的所有图片名
        features[i] = image.imread('%s/image/%s.jpg' % (root, fname))
        labels[i] = image.imread('%s/label/%s.png' % (root, fname))


    return features, labels


def tct_resize(feature, label, height, width):
    feature_ = image.imresize(feature, width, height)
    label_ = image.imresize(label, width, height)
    return feature_, label_


class TCTSegDataset(gdata.Dataset):
    def __init__(self, tct_dir, is_train):
        self.is_train = is_train
        
        self.rgb_mean = nd.array([0.5228142, 0.54256412, 0.5228142])
        self.rgb_std = nd.array([0.20718039, 0.20099298, 0.23153676])
        

        self.resize_size = 512
        features, labels = read_person_data(root=tct_dir, is_train=is_train)
        self.features = [self.normalize_image(feature) for feature in features]
        self.labels = labels

    def __getitem__(self, idx):
        if self.is_train:
            feature, label = tct_resize(self.features[idx], self.labels[idx], self.resize_size, self.resize_size)
        else:
            feature, label = tct_resize(self.features[idx], self.labels[idx], self.resize_size, self.resize_size)
        return feature.transpose((2, 0, 1)), label.transpose((2, 0, 1))

    def __len__(self):
        return len(self.features)

    def normalize_image(self, img):
        # print('before normalize:', img)
        # while(True):
        #     cv2.imshow("outputs", img)  # 输出一张400*400的白色图片(255 255 255):蓝(B)、绿(G)、红(R)
        #     if cv2.waitKey(10) & 0xFF == ord('q'):
        #         break
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

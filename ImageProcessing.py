import tensorflow as tf
import numpy as np
import os
import cv2
import PIL.Image

train_data_num = 1500
test_data_num = 600
max_sequence_length = 5

class dataprocessing():
    def __init__(self,train_or_test,target_height = 224,target_width = 224):
        self.target_width = target_width
        self.target_height = target_height
        all_samples = []
        samples_num = 0
        data_path = None
        self.begin_pos = 0
        start_sample = 1
        if train_or_test == 'train':
            samples_num += train_data_num
            data_path = './data/amap_traffic_train_0712'
        elif train_or_test == 'test':
            samples_num += test_data_num
            data_path = './data/amap_traffic_test_0712'

        padding_datasets = np.zeros([samples_num,max_sequence_length,target_height,target_width,3])
        sample_num = 0
        sequence_length = []
        for sample in os.listdir(data_path):
            if sample.startswith('.'):
                continue
            per_sample = []
            for per_picture in os.listdir(data_path + '/' + sample):
                img = cv2.imread(data_path + '/' + sample + '/' + per_picture)
                # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = self.adjust(img)
                img = img / 255.0
                per_sample.append(img)

            padding_datasets[sample_num,:len(per_sample),:,:,:] = per_sample
            sample_num += 1
            sequence_length.append(len(per_sample))

        self.padding_datasets = padding_datasets  # samples_nums*pic_num*h*w*c
        self.sequence_length = sequence_length

    def next_batch_samples(self,batch_size):
        result = self.padding_datasets[self.begin_pos:self.begin_pos + batch_size]
        self.begin_pos += batch_size
        return result

    def next_batch_sequence_length(self,batch_size):
        result = self.sequence_length[self.begin_pos:self.begin_pos + batch_size]
        self.begin_pos += batch_size
        return result

    @property
    def samples_num(self):
        return len(self.padding_datasets)


    def adjust(self, img):
        wn, wh = self.__get_slice(img.shape[1], img.shape[0])
        img = self.__slice_img(img, wn, wh)
        img = self.__resize(img)
        return img

    # 1：计算原图应该留下的部分
    def __get_slice(self, _width, _height):
        # 比较哪个缩放倍数更小
        wscale = _width / self.target_width
        hscale = _height / self.target_height
        if wscale < hscale:
            scale = wscale
        else:
            scale = hscale
        # 四舍五入到整数
        wn = round(scale * self.target_width)
        wh = round(scale * self.target_height)
        return wn, wh

    # 2：原图居中，切割掉多余的边缘部分
    def __slice_img(self, img, slicew, sliceh):
        midw = img.shape[1] / 2
        midh = img.shape[0] / 2
        wl = round(midw - slicew / 2)
        wh = round(midw + slicew / 2)
        hl = round(midh - sliceh / 2)
        hh = round(midh + sliceh / 2)
        return img[hl:hh, wl:wh, :]

    # 3: 缩放
    def __resize(self, img):
        return cv2.resize(img, (self.target_width, self.target_height))




if __name__ == '__main__':
    aaa = dataprocessing('test')
    print(aaa.next_batch_sequence_length(32))
    print(11)



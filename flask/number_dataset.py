import numpy as np
import random
import os
from PIL import Image

class dataset:

    # 图片的名称集合
    files_name = []
    # 图片集合大小
    files_length = 0
    # 图片名称的集合指针
    filename_ptr = 0
    # 图片所在位置
    data_location = ''
    # 验证码的长度
    captcha_length = 4
    # 字符集
    captcha_set_length = 10
    # 图片宽度
    image_width = 160
    # 图片高度
    image_height = 60


    def __init__(self, data_location):
        self.data_location = data_location
        if not os.path.exists(data_location):
            raise RuntimeError('不存在图片路径')
        self.files_name = os.listdir(data_location)
        self.files_length = len(self.files_name)
        random.shuffle(self.files_name)

    def next_batch(self, batch_size, file_name = False):

        batch_x = np.zeros([batch_size, self.image_height * self.image_width])
        batch_y = np.zeros([batch_size, self.captcha_length * self.captcha_set_length])

        self.filename_ptr = random.randint(0, self.files_length-1)

        if self.filename_ptr + batch_size < self.files_length:
            image_batch = self.files_name[self.filename_ptr : self.filename_ptr + batch_size]
            label_batch = list(map(lambda x : x[-8:-4], image_batch))
            # self.filename_ptr = self.filename_ptr + batch_size
        else:
            new_ptr = (self.filename_ptr + batch_size) % self.files_length
            image_batch = self.files_name[self.filename_ptr:] + self.files_name[:new_ptr]
            label_batch = list(map(lambda x : x[-8:-4], image_batch))
            # random.shuffle(self.files_name)
            # self.filename_ptr = new_ptr

        for index, image in enumerate(image_batch):
            image_grey = np.array(Image.open(self.data_location + image).convert('L'), 'f') / 255
            batch_x[index,:] = image_grey.flatten()
        for index, label in enumerate(label_batch):
            batch_y[index,:] = self.decode_one_hot(label)
        if file_name == False:
            return batch_x, batch_y
        else:
            return batch_x, batch_y, self.data_location, image_batch
    
    
    def decode_one_hot(self, text):
        vector = np.zeros(self.captcha_length * self.captcha_set_length)
        def char2pos(c):
            if ord(c) >= 48 and ord(c) <= 57:
                return ord(c) - 48
            if ord(c) >= 65 and ord(c) <= 90:
                return ord(c) - 55
            if ord(c) >= 97 and ord(c) <= 122:
                return ord(c) - 87
        for i, c in enumerate(text):
            idx = i * self.captcha_set_length + char2pos(c)
            vector[idx] = 1
        return vector
 
    def encode_one_hot(self, vec):
        char_pos = vec.nonzero()[0]
        text = []
        for i,c in enumerate(char_pos):
            char_index = c % self.captcha_set_length
            if char_index < 10:
                text.append(chr(char_index + ord('0')))
            else:
                text.append(chr(char_index - 10 + ord('a')))
        # return text
        return ''.join(text)

if __name__ == '__main__':
    dataset = dataset('./data/test1/')
    batch_x, batch_y = dataset.next_batch(3)
    print(batch_x,)
    print(batch_y,)


    

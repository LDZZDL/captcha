import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import number_generate_captcha as ngc
from number_captcha_cnn import captcha
from number_dataset import dataset

def captcha_number():

    result = {}
    file_name = []
    captcha_text = []
    dir = ngc.random_text(ngc.Lowercase+ngc.Uppercase+ngc.Number,7)
    result['dir'] = dir
    dir = './static/data/' + dir + '/'
    os.makedirs(dir)
    for _ in range(0,8):
        a,b = ngc.gen_captcha(dir)
        captcha_text.append(a)
        file_name.append(b)
    result['file'] = file_name
    result['captcha'] = captcha_text
    return json.dumps(result)



def recognize(dir):
    data_location = './static/data/' + dir + '/'
    captcha_class = captcha(data_location)
    result = captcha_class.recognize_number('./model/number/')
    return json.dumps(result)

if __name__ == '__main__':
    json = captcha_number()
    print(json)


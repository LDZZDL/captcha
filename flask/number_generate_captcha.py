import numpy as np
from PIL import Image
import random
import sys
import os
from captcha.image import ImageCaptcha

Number = [str(i) for i in range(10)]
Lowercase  = [chr(i) for i in range(97,123)]
Uppercase  = [chr(i) for i in range(65,91)]



# print(CHAR_SET)

# 随机生成4个字符
def random_text(char_set=Number,captcha_size=4):
    return ''.join(np.random.choice(char_set,captcha_size))

# 生成字符对应的验证码
def gen_captcha_text_and_image(path):
    image_ = ImageCaptcha()
    captcha_text = random_text()
    image_.generate(captcha_text)
    image_.write(captcha_text, 
        path + random_text(char_set=Number + Lowercase + Uppercase,captcha_size=5) + '_' + captcha_text + '.jpg')
# 
def gen_captcha(path):
    image_ = ImageCaptcha()
    captcha_text = random_text()
    image_.generate(captcha_text)
    file_name = path + random_text(char_set=Number + Lowercase + Uppercase,captcha_size=5) + '_' + captcha_text + '.jpg'
    image_.write(captcha_text, file_name)
    return captcha_text,file_name

#path = './data/train/'
path = './'
num = 10
if __name__ == '__main__':
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(num):
        gen_captcha_text_and_image(path)
        sys.stdout.write('\r>> Creating image %d/%d' % (i+1,num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
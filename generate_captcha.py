import numpy as np
from PIL import Image
import random
import sys
import os
from captcha.image import ImageCaptcha

Number = [str(i) for i in range(10)]
Lowercase  = [chr(i) for i in range(97,123)]
Uppercase  = [chr(i) for i in range(65,91)]

# 设置验证码图片生成的路径
path = './'
# 设置验证码生成的数量
num = 10
# 设置验证码锁用到的字符集合
# 数字
set = Number
# 数字+大小写数字混合验证码
# set = Number + Lowercase + Uppercase

# 随机生成4位数字字符
def random_text(char_set=set,captcha_size=4):
    return ''.join(np.random.choice(char_set,captcha_size))

# 生成字符对应的验证码
def gen_captcha_text_and_image(path):
    image_ = ImageCaptcha()
    # 获取生成的验证码
    captcha_text = random_text()
    image_.generate(captcha_text)
    # 验证码图片名称的前缀，防止相同验证码互相覆盖
    prefix = random_text(char_set=Number + Lowercase + Uppercase,captcha_size=5) + '_'
    image_.write(captcha_text, path + prefix + captcha_text + '.jpg')

if __name__ == '__main__':
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(num):
        gen_captcha_text_and_image(path)
        sys.stdout.write('\r>> Creating image %d/%d' % (i+1,num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
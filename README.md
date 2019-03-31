# 使用TensorFlow构建CNN网络破解验证码

## 验证码
既然要使用CNN来训练一个破解验证码的模型，我们首先需要去获标注好的验证码图像——即已经说明了图像验证码中验证码的确切值。我们使用的是Python中的[captcha库](https://github.com/lepture/captcha)来生成验证码图片，图片大小为160*60，文件名为前缀+验证码值。
![](./images/00dlF_8080.jpg)
![](./iamges/00Gni_1482.jpg)
![](./images/00lET_7852.jpg)
![](./images/Snipaste_2019-03-31_11-27-08.png)

可以在[generate_captcha.py](./generate_captcha.py)文件中修改我们需要生成验证码所需要的参数。

——有人要问为什么不去获取真实的验证码数据？
——获取真实的验证码图片容易，但是你需要对其进行标注，就很困难。比如你爬取了10万张验证码图片，但是你需要手工标注出这10万张图片的内容。

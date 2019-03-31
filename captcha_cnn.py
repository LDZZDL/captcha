import tensorflow as tf
import numpy as np
from dataset import dataset
from PIL import Image


class captcha:

    # 日志输出路径
    train_log_path = './logs/'

    x = tf.placeholder(tf.float32, [None, dataset.image_height * dataset.image_width])
    y = tf.placeholder(tf.float32, [None, dataset.captcha_length * dataset.captcha_set_length])
    keep_prob = tf.placeholder(tf.float32)

    def __init__(self, data_location):
        self.dataset = dataset(data_location)
        
    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    

    def captcha_cnn(self, w_alpha= 0.01, b_alpha= 0.01):
        x_image = tf.reshape(self.x, shape=[-1,self.dataset.image_height, self.dataset.image_width, 1])
        # 卷积层
        w_c1 = tf.Variable(w_alpha*tf.random_normal([3,3,1,16]))
        tf.add_to_collection('l2', tf.contrib.layers.l2_regularizer(0.001)(w_c1))
        b_c1 = tf.Variable(b_alpha*tf.random_normal([16]))
        conv1 = tf.nn.relu(tf.nn.bias_add(self.conv2d(x_image, w_c1),b_c1))
        # 池化层
        pool1 = self.max_pool_2x2(conv1)
        # 卷积层
        w_c2 = tf.Variable(w_alpha*tf.random_normal([3,3,16,32]))
        tf.add_to_collection('l2', tf.contrib.layers.l2_regularizer(0.001)(w_c2))
        b_c2 = tf.Variable(b_alpha*tf.random_normal([32]))
        conv2 = tf.nn.relu(tf.nn.bias_add(self.conv2d(pool1,w_c2), b_c2))
        # 池化层
        pool2 = self.max_pool_2x2(conv2)

        # 卷积层
        w_c3 = tf.Variable(w_alpha*tf.random_normal([3,3,32,64]))
        tf.add_to_collection('l2', tf.contrib.layers.l2_regularizer(0.001)(w_c3))
        b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
        conv3 = tf.nn.relu(tf.nn.bias_add(self.conv2d(pool2,w_c3), b_c3))
        #  池化层
        pool3 = self.max_pool_2x2(conv3)

        # 全连接层
        w_fc1 = tf.Variable(w_alpha*tf.random_normal([20*8*64, 1024]))
        tf.add_to_collection('l2', tf.contrib.layers.l2_regularizer(0.001)(w_fc1))
        b_fc1 = tf.Variable(b_alpha*tf.random_normal([1024]))
        pool3_flat = tf.reshape(pool3, [-1, 20*8*64])
        fc1 = tf.nn.relu(tf.add(tf.matmul(pool3_flat,w_fc1), b_fc1))
        fc1_dropout = tf.nn.dropout(fc1, self.keep_prob)
        # 全连接层
        w_fc2 = tf.Variable(w_alpha*tf.random_normal([1024, self.dataset.captcha_length * self.dataset.captcha_set_length]))
        tf.add_to_collection('l2', tf.contrib.layers.l2_regularizer(0.001)(w_fc2))
        b_fc2 = tf.Variable(b_alpha*tf.random_normal([self.dataset.captcha_length * self.dataset.captcha_set_length]))
        fc2 = tf.matmul(fc1_dropout,w_fc2) + b_fc2

        return fc2

    # 训练模型
    def train_captcha_cnn(self):
        out = self.captcha_cnn()
        #loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(out)))
        loss = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=self.y)))
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
        
        tf.summary.scalar('loss', loss)

        
        prediction = tf.reshape(out, [-1, self.dataset.captcha_length, self.dataset.captcha_set_length])
        label = tf.reshape(self.y, [-1, self.dataset.captcha_length, self.dataset.captcha_set_length])
        correct_pred = tf.equal(tf.argmax(prediction, 2), tf.argmax(label, 2))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(self.train_log_path + 'train', sess.graph)
            test_writer = tf.summary.FileWriter(self.train_log_path + 'test')
            sess.run(tf.global_variables_initializer())
            for i in range(50000):
                if i % 10 == 0:
                    batch_x_test, batch_y_test = self.dataset.next_batch(100)
                    summmary, acc = sess.run([merged, accuracy], feed_dict={self.x:batch_x_test,self.y:batch_y_test,self.keep_prob:1.0})
                    print('迭代第%d次 accuracy:%f' % (i+1, acc))
                    test_writer.add_summary(summmary, i)
                    #if acc > 0.99:
                    #    train_writer.close()
                    #    test_writer.close()
                    #    saver.save(sess, './model/crack_capcha.model', global_step=i)
                    #    break
                else:
                    batch_x_train, batch_y_train = self.dataset.next_batch(100)
                    loss_val, _ = sess.run([loss, optimizer], feed_dict={self.x:batch_x_train,self.y:batch_y_train,self.keep_prob:0.7})
                    print('迭代第%d次 loss:%f' % (i+1, loss_val))
                    curve = sess.run(merged, feed_dict={self.x:batch_x_train,self.y:batch_y_train,self.keep_prob:0.7})
                    train_writer.add_summary(curve, i)
            train_writer.close()
            test_writer.close()
            saver.save(sess, './model/crack_capcha.model', global_step=50000)

    # 测试模型
    def test_captcha_cnn(self):
        out = self.captcha_cnn()
        saver = tf.train.Saver()
        correct = 0
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('./model/'))
            # batch_x, batch_y, file_path, files_name = self.dataset.next_batch(10, True)
            batch_x, batch_y = self.dataset.next_batch(100)
            y_ = sess.run(out, feed_dict={self.x: batch_x, self.keep_prob: 1})
            y__ = sess.run(tf.argmax(tf.reshape(y_, [-1, self.dataset.captcha_length, self.dataset.captcha_set_length]), 2))
            
            for i in range(100):
                vector = np.zeros(self.dataset.captcha_length*self.dataset.captcha_set_length)
                j = 0
                for n in (y__[i].tolist()):
                        vector[j*self.dataset.captcha_set_length + n] = 1
                        j += 1
                prediction_text = self.dataset.encode_one_hot(vector)
                label = self.dataset.encode_one_hot(batch_y[i])
                print('正确:',label, '  预测:',prediction_text)
                if prediction_text == label:
                    correct = correct + 1
                # show_image(file_path, files_name[i])
        print('correct:',correct)

    # 调用模型，来识别图片
    def recognize_number(self,model_path):
        out = self.captcha_cnn()
        saver = tf.train.Saver()
        correct = 0
        result = {}
        correct_result = []
        prediction_result = []
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            # batch_x, batch_y, file_path, files_name = self.dataset.next_batch(10, True)
            for _ in range(1,10):
                batch_x, batch_y = self.dataset.next_batch(100)
                y_ = sess.run(out, feed_dict={self.x: batch_x, self.keep_prob: 1})
                y__ = sess.run(tf.argmax(tf.reshape(y_, [-1, self.dataset.captcha_length, self.dataset.captcha_set_length]), 2))
            
                for i in range(100):
                    vector = np.zeros(self.dataset.captcha_length*self.dataset.captcha_set_length)
                    j = 0
                    for n in (y__[i].tolist()):
                        vector[j*self.dataset.captcha_set_length + n] = 1
                        j += 1
                    prediction_text = self.dataset.encode_one_hot(vector)
                    label = self.dataset.encode_one_hot(batch_y[i])
                    print('正确:',label, '  预测:',prediction_text)
                    correct_result.append(label)
                    prediction_result.append(prediction_text)
                    if prediction_text == label:
                        correct = correct + 1
        result['correct'] = correct
        result['correct_result'] = correct_result
        result['prediction_result'] = prediction_result
        print(correct)
        return result

if __name__ == '__main__':
    # D:\\Code\\TF\\data\\test1\\
    captcha = captcha('D:\\Code\\TensorFlow-Train\\data-train\\')
    captcha.train_captcha_cnn()

    #captcha = captcha('D:\\Code\\TensorFlow-Train\\data-test\\')
    #captcha.recognize_number('./model/')

    
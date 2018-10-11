import tensorflow as tf
from util import tf_utils, processing
import pprint
import numpy as np
from Network.GAN import LSGAN

class CGAN(LSGAN):
    def __init__(self, input_shape, learning_rate, noise_dim, num_class=1,num_labels = 10, sess=None, ckpt_path=None, net='cgan'):
        super().__init__(input_shape, learning_rate, noise_dim, num_class, sess, ckpt_path, net)
        self.num_labels = num_labels

    def __build_net__(self):
        self.X = tf.placeholder(shape=[None]+self.input_shape, dtype=tf.float32, name='X')
        self.Z = tf.placeholder(shape=[None, self.noise_dim], dtype=tf.float32, name='random_z')
        self.y = tf.placeholder(shape=[None], dtype=tf.uint8, name='input_Y')
        self.Y = tf.one_hot(indices=self.y, depth=self.num_labels, dtype=tf.float32, axis=-1)

        self.G = self.Generator(tf.concat([self.Z,self.Y],-1) , self.X.shape[1])

        self.D = self.Discriminator(tf.concat([self.X,self.Y],-1), self.num_classes)
        self.D_G = self.Discriminator(tf.concat([self.G,self.Y],-1), self.num_classes)

        self.__set_loss_and_optim__()

        return

    def train(self, y= None, x= None, z=None):
        if x is None:
            return self.sess.run([self.G_optim, self.G_loss], feed_dict = {
                self.Z: z,
                self.y: y
            })[1]
        else:
            x = processing.img_preprocessing(x)
            return self.sess.run([self.D_optim, self.D_loss], feed_dict = {
                self.X: x,
                self.Z: z,
                self.y: y
            })[1]

    def eval(self, z, y=None):
        out = self.sess.run(self.G, feed_dict = {self.Z: z, self.y: y})
        out = processing.img_deprocessing(out)
        return processing.show_images(out)


    def infer(self, z, y=None, path = '../generated/save.png'):
        fig = self.eval(z, y)
        return fig.savefig(path)
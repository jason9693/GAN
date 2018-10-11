import tensorflow as tf
from util import tf_utils, processing
import pprint
import numpy as np
import params as par

class GAN:
    def __init__(self, input_shape, learning_rate,noise_dim, num_classes=1, sess=None, ckpt_path=None, net='gan'):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.sess= sess
        self.noise_dim = noise_dim
        self.net = net
        self.__build_net__()

        if ckpt_path is not None:
            saver = tf.train.Saver()
            saver.restore(sess=self.sess, save_path=ckpt_path)
        else:
            self.sess.run(tf.global_variables_initializer())

    def __build_net__(self):
        self.X = tf.placeholder(shape=[None]+self.input_shape, dtype=tf.float32, name='X')
        self.Z = tf.placeholder(shape=[None, self.noise_dim], dtype=tf.float32, name='random_z')

        self.G = self.Generator(self.Z , self.X.shape[1])
        #tf.random_normal(shape=[self.batch_size]+self.input_shape, dtype=tf.float32)

        self.D = self.Discriminator(self.X, self.num_classes)
        self.D_G = self.Discriminator(self.G, self.num_classes)

        self.__set_loss_and_optim__()
        return

    def Discriminator(self, input, output_dim, name = 'discriminator'):
        with tf.variable_scope('discriminator',reuse=tf.AUTO_REUSE):
            L1 = tf_utils.Dense(input, 256, name=name+'/L1', activation=tf.nn.leaky_relu)
            L2 = tf_utils.Dense(L1, 256, name=name+'/L2', activation=tf.nn.leaky_relu)
            L3 = tf_utils.Dense(L2, output_dim, name=name+'/L3', activation=None)
        return L3

    def Generator(self,z , output_dim, name= 'generator'):
        #gaussian = tf.random_uniform(shape=[self.batch_size]+[input_dim], dtype=tf.float32, minval=-1, maxval=1)
        L1 = tf_utils.Dense(z, z.shape[1] // 2, name=name+'/L1', activation=tf.nn.relu)
        L2 = tf_utils.Dense(L1, z.shape[1], name=name + '/L2', activation=tf.nn.relu)
        L3 = tf_utils.Dense(L2, output_dim, name=name + '/L3', activation=None)
        return tf.tanh(L3)

    def __set_loss_and_optim__(self):
        logits_real = tf.ones_like(self.D_G)
        logits_fake = tf.zeros_like(self.D_G)

        self.G_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_G, labels=logits_real)
        self.G_loss = tf.reduce_mean(self.G_loss)

        self.D_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D, labels=logits_real) \
        + tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_G, labels=logits_fake)
        self.D_loss = tf.reduce_mean(self.D_loss)

        D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

        print('Discriminator variables: ',D_vars)

        print('\nGenerator variables: ', G_vars)

        self.D_optim = \
            tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.D_loss, var_list=D_vars)
        self.G_optim = \
            tf.train.AdamOptimizer(self.learning_rate).minimize(self.G_loss, var_list=G_vars)
        return

    def train(self, x= None, z=None, y=None):
        if x is None:
            return self.sess.run([self.G_optim, self.G_loss], feed_dict = {
               self.Z: z
            })[1]
        else:
            x = processing.img_preprocessing(x)
            return self.sess.run([self.D_optim, self.D_loss], feed_dict = {
                self.X: x,
                self.Z: z
            })[1]

    def eval(self, z, y=None):
        out = self.sess.run(self.G, feed_dict = {self.Z: z})
        out = processing.img_deprocessing(out)
        fig = processing.show_images(out,'generated/save.png')

        return fig

    def infer(self, z, y=None, path=None):
        fig = self.eval(z)
        fig.savefig('generated/{}.png'.format(self.net))
        return


class LSGAN(GAN):
    def __init__(self, input_shape, learning_rate, noise_dim, num_classes=1, sess=None, ckpt_path=None, net='lsgan'):
        super().__init__(input_shape, learning_rate, noise_dim, num_classes, sess, ckpt_path, net)

    def __set_loss_and_optim__(self):
        self.G_loss = tf.square(tf.nn.sigmoid(self.D_G) - 1)
        self.G_loss = tf.reduce_mean(self.G_loss)

        self.D_loss = tf.square(self.D - 1) + tf.square(self.D_G)
        self.D_loss = tf.reduce_mean(self.D_loss)

        D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

        self.D_optim = \
            tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.D_loss, var_list=D_vars)
        self.G_optim = \
            tf.train.AdamOptimizer(self.learning_rate).minimize(self.G_loss, var_list=G_vars)

        print('Discriminator variables: {}, loss: {}'.format(D_vars, self.D_loss))
        print('\nGenerator variables: {}, loss: {}'.format(G_vars, self.D_loss))

        return

class DCGAN(GAN):
    def __init__(self, input_shape, learning_rate,noise_dim, num_classes=1, sess=None, ckpt_path=None, net='DCGAN'):
        self.init_kernel_size = 7
        self.init_filter_size = 512
        super().__init__(input_shape, learning_rate, noise_dim, num_classes, sess, ckpt_path, net)
    #
    # def __set_loss_and_optim__(self):
    #     logits_real = tf.ones_like(self.D_G)
    #     logits_fake = tf.zeros_like(self.D_G)
    #
    #     self.G_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_G, labels=logits_real)
    #     self.G_loss = tf.reduce_mean(self.G_loss)
    #
    #     self.D_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D, labels=logits_real) \
    #     + tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_G, labels=logits_fake)
    #     self.D_loss = tf.reduce_mean(self.D_loss)
    #
    #     D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    #     G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
    #
    #     print('Discriminator variables: ',D_vars)
    #
    #     print('\nGenerator variables: ', G_vars)
    #
    #     self.D_optim = \
    #         tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.D_loss, var_list=D_vars)
    #     self.G_optim = \
    #         tf.train.AdamOptimizer(self.learning_rate).minimize(self.G_loss, var_list=G_vars)
    #     return

    def Generator(self, z, output_dim, name= 'generator'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) and tf_utils.set_device_mode(par.gpu_mode):
            l0 = tf_utils.Dense(
                z,
                self.init_filter_size * self.init_kernel_size * self.init_kernel_size,
                'l0',
                activation=tf.nn.relu
            )
            l0 = tf.reshape(l0, [-1, self.init_kernel_size, self.init_kernel_size, self.init_filter_size])
            l0 = tf.layers.batch_normalization(l0)

            l1 = tf.layers.conv2d_transpose(
                l0,
                self.init_filter_size//2,
                kernel_size=[5,5],
                strides=(2,2),
                padding='same',
                activation=tf.nn.relu
            )
            l1 = tf.layers.batch_normalization(l1)

            final_layer = tf.layers.conv2d_transpose(l1, 1, [5,5], strides=(2,2), padding='same', activation=tf.nn.tanh)
            print(final_layer.shape)
            return tf.layers.flatten(final_layer)

    def Discriminator(self, input, output_dim, name = 'discriminator'):
        with tf.variable_scope(name,reuse= tf.AUTO_REUSE) and tf_utils.set_device_mode(par.gpu_mode):
            input = tf.reshape(input, [-1, 28,28,1])

            l0 = tf.layers.conv2d(input, 1, [5,5], strides=(1,1), padding='same')

            l1 = tf.layers.conv2d(l0, self.init_filter_size//4, [5,5], strides=(2,2), padding='same', activation=tf.nn.leaky_relu)
            l1 = tf.layers.batch_normalization(l1)

            l2 = tf.layers.conv2d(
                l1,
                self.init_filter_size//2,
                [5,5],
                strides=(2,2),
                padding='same',
                activation=tf.nn.leaky_relu
            )
            l2 = tf.layers.batch_normalization(l2)

            l3 = tf.layers.flatten(l2)
            l3 = tf_utils.Dense(l3, 64, 'l3', activation=tf.nn.leaky_relu)

            logits = tf_utils.Dense(l3, output_dim, 'logits')

            return logits

        #tf.reshape(z, [z.shape[0], 1, tf.sqrt(z.shape[1]),tf.sqrt(z.shape[1])])
        # conv1 = tf.layers.conv2d(z, 1024, [4,4])
class WGAN(DCGAN):
    def __init__(self, input_shape, learning_rate,noise_dim, num_classes=1, sess=None, ckpt_path=None, net='WGAN'):
        self.init_kernel_size = 7
        self.init_filter_size = 512
        super().__init__(input_shape, learning_rate, noise_dim, num_classes, sess, ckpt_path, net)

    def __set_loss_and_optim__(self):
        # self.G_loss = tf.nn.sigmoid(self.D_G)
        self.G_loss = - tf.reduce_mean(self.D_G)

        self.D_loss = - self.D + self.D_G
        self.D_loss = tf.reduce_mean(self.D_loss)

        D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

        print('Discriminator variables: ', D_vars)
        print('\nGenerator variables: ', G_vars)

        self.D_optim = \
            tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.D_loss, var_list=D_vars)
        self.G_optim = \
            tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.G_loss, var_list=G_vars)

        d_clip = [var.assign(tf.clip_by_value(var,-0.1, 0.1)) for var in D_vars]
        self.D_optim = (self.D_optim, d_clip)
        return


class WGAN_GP(WGAN):
    def __init__(self, input_shape, learning_rate,noise_dim, num_classes=1, sess=None, ckpt_path=None, net='WGANGP'):
        self.init_kernel_size = 7
        self.init_filter_size = 512
        super().__init__(input_shape, learning_rate, noise_dim, num_classes, sess, ckpt_path, net)
#%%

import matplotlib.pyplot as plt

import os
import numpy
import skimage.io
import scipy
import skimage.transform
import skimage.color

os.chdir('/home/bbales2/dic')

im = skimage.color.rgb2gray(skimage.io.imread('DIC_example.tif').astype('float'))#'images/cho01.png'

im = skimage.transform.rescale(im, 0.25)

im -= im.flatten().min()
im /= im.flatten().max()

im -= numpy.mean(im.flatten())

plt.imshow(im, cmap = plt.cm.gray)
plt.colorbar()
plt.show()

#%%

import tensorflow as tf
sess = tf.InteractiveSession()

#%%

GradCpu = numpy.concatenate((numpy.array([[0, 0, 0],
                               [-0.5, 0, 0.5],
                               [0, 0, 0]]).reshape((3, 3, 1, 1)),
                  numpy.array([[0, -0.5, 0],
                               [0, 0, 0],
                               [0, 0.5, 0]]).reshape((3, 3, 1, 1))), axis = 3).astype('float32')

kernelCpu = scipy.ndimage.filters.gaussian_filter([[0.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0]], 0.75).reshape((3, 3, 1, 1)).astype('float32')

kernelCpu /= kernelCpu.flatten().sum()

Grad = tf.constant(GradCpu)
kernel = tf.constant(kernelCpu)

u = tf.Variable(tf.random_uniform(shape = [2, 1], minval = 0.0, maxval = 1.0))

I = tf.Variable(tf.truncated_normal(shape = [1, im.shape[0], im.shape[1], 1], mean = 0.0, stddev = 0.1))

#%%

G = tf.placeholder(tf.float32, shape = im.shape)

#%%
blur = tf.nn.conv2d(I, kernel, strides=[1, 1, 1, 1], padding='SAME')
conv = tf.nn.conv2d(blur, Grad, strides=[1, 1, 1, 1], padding='SAME')
Gt = tf.reshape(tf.matmul(tf.reshape(conv, [-1, 2]), u), im.shape)

Gloss = tf.nn.l2_loss(Gt - G)
Rloss = 0.05 * tf.reduce_sum(tf.abs(conv))
uloss = tf.nn.l2_loss(tf.reduce_sum(u * u) - 1)

tloss = Gloss + Rloss + uloss

dtrain_step = tf.train.AdamOptimizer(1e-2).minimize(tloss, var_list = [I, u])

#%%
sess.run(tf.initialize_all_variables())

batch_size = 50
ces = []
for i in range(10000):
    if i % 10 == 0 and i > 0:
        plt.plot(ces)
        plt.show()
        tmp = I.eval()[0, :, :, 0]
        plt.imshow(tmp, interpolation = 'NONE', cmap = plt.cm.gray)
        plt.colorbar()
        plt.show()
        tmp = blur.eval()[0, :, :, 0]
        plt.imshow(tmp, interpolation = 'NONE', cmap = plt.cm.gray)
        plt.colorbar()
        plt.show()
        tmp = Gt.eval()[:, :]
        plt.imshow(tmp, interpolation = 'NONE', cmap = plt.cm.gray)
        plt.colorbar()
        plt.show()
        print 'fwd', tlosst, dl1, dl2

    train, tlosst, dl1, dl2 = sess.run([dtrain_step, tloss, Gloss, Rloss], feed_dict = { G : im })

    print tlosst, dl1, dl2
    ces.append(tlosst)

#%%
output = blur.eval()[0, :, :, 0]
output -= output.flatten().min()
output /= output.flatten().max()
skimage.io.imsave('dic_processed.png', output)
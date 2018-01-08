import tensorflow as tf
from model import Model
from dataloader import Dataset
from utils import *
import numpy as np
import os

class Test:
    def __init__(self, batch_size, data_path, model_path, output_path):
        self.epoch = 1
        self.batch_size = batch_size
        self.model = Model(batch_size)
        self.test_dataloader = Dataset(os.path.join(data_path, 'test'))
        self.model_path = model_path
        self.output_path = output_path
        
    def test(self):
        idx_test = 0
        block_mask, inverse_block_mask = create_block_mask(shape=[self.batch_size, 96, 96, 3])
        
        with tf.device('/gpu:0'):
            with tf.Graph().as_default():
                #input
                random = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 96, 96, 3])
                y = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 96, 96, 3])
                mask = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 96, 96, 3])
                inverse_mask = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 96, 96, 3])
                x = y*mask + random*inverse_mask
                
                #generator
                G_output_sample = self.model.generator(x)
                
                config = tf.ConfigProto()
                config.gpu_options.per_process_gpu_memory_fraction = 0.6
                
                with tf.Session(config=config) as sess:
                    sess.run(tf.global_variables_initializer())
                    restorer = tf.train.Saver()
                    try:
                        restorer.restore(sess, tf.train.latest_checkpoint(self.model_path))
                        print('Load model Success')
                    except:
                        print('No model to restore ...')
                        raise
                    
                    print('Start testing ...')
                    count = 0
                    while True:
                        epoch, idx_test, y_batch = self.test_dataloader.load_batch(self.batch_size, idx_test, size=[96, 96])
                        random_noise = np.random.normal(size=y_batch.shape)
                        G_output_out, x_batch = sess.run([G_output_sample, x], feed_dict={random:random_noise, y:y_batch, mask:block_mask, inverse_mask:inverse_block_mask})
                        
                        #visualize
                        G_output_out = G_output_out*inverse_block_mask + y_batch*block_mask
                        G_output_out = np.squeeze(G_output_out)
                        print('Saving image {0}' .format(str(count)+'-noiseX-test'))
                        plot(x_batch, name=str(count)+'-noiseX-test' ,output_path=self.output_path)
                        print('Saving image {0}' .format(str(count)+'-realX-test'))
                        plot(y_batch, name=str(count)+'-realX-test' ,output_path=self.output_path)
                        print('Saving image {0}' .format(str(count)+'-fakeG-test'))
                        plot(G_output_out, name=str(count)+'-fakeG-test' ,output_path=self.output_path)
                        count += 1
                        if epoch == self.epoch:
                            print('Finish ...')
                            break


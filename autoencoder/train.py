import tensorflow as tf
from model import Model
from dataloader import Dataset
from utils import *
import numpy as np
import os


class Train:
    def __init__(self, epoch, batch_size, data_path, model_path, output_path, graph_path, restore=False):
        self.batch_size = batch_size
        self.model = Model(batch_size)
        self.train_dataloader = Dataset(os.path.join(data_path, 'train'))
        self.train_test_dataloader = Dataset(os.path.join(data_path, 'train'))
        self.test_dataloader = Dataset(os.path.join(data_path, 'test'))
        self.epoch = epoch
        self.model_path = model_path
        self.output_path = output_path
        self.graph_path = graph_path
        self.restore = restore
      
    def train(self):
        idx_train = 0
        idx_train_test = 0
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
                G_output = self.model.generator(x)
                G_output_sample = self.model.generator(x, reuse=True)
            
                G_output = G_output*inverse_mask + y*mask
                
                #loss         
                texture_loss = tf.nn.l2_loss(y*mask-G_output*mask)
                crop_loss = tf.nn.l2_loss(y*inverse_mask-G_output*inverse_mask)
                X_loss = texture_loss + crop_loss

                global_step = tf.Variable(0, trainable=False)
                #optimizer
                X_op = tf.train.AdamOptimizer(learning_rate=1e-4)
                X_train_op = X_op.minimize(X_loss, global_step=global_step)          
    
                #tensorboard
                tf.summary.scalar('X_loss', X_loss)
                summary_op = tf.summary.merge_all()
                
                #GPU ratio
                config = tf.ConfigProto()
                config.gpu_options.per_process_gpu_memory_fraction = 0.6
                
                with tf.Session(config=config) as sess:
                    sess.run(tf.global_variables_initializer())
                    saver = tf.train.Saver(max_to_keep=5)
                    if self.restore:
                        try:
                            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
                            print('Load model Success')
                        except:
                            print('No model to restore ...')
                            raise
                    summary_writer = tf.summary.FileWriter(self.graph_path, sess.graph)
                    print('Start training ...')
                    
                    while True:
                        epoch, idx_train, y_batch = self.train_dataloader.load_batch(self.batch_size, idx_train, size=[96, 96])
                        random_noise = np.random.normal(size=y_batch.shape)
                       
                        _, loss_X, step, summary = sess.run([ X_train_op, X_loss, global_step, summary_op],
                                                                feed_dict={random:random_noise, y:y_batch, mask:block_mask, inverse_mask:inverse_block_mask})
                        summary_writer.add_summary(summary, global_step=step)
                        if step % 10 == 0:
                            print('Epoch: {0} Step: {1} Loss_X: {2}'.format(epoch, step, loss_X))
                               
                        #sample 
                        if step % 100 == 0:    
                            #sample form training data
                            _, idx_test, y_batch = self.test_dataloader.load_batch(self.batch_size, idx_test, size=[96, 96])
                            random_noise = np.random.normal(size=y_batch.shape)
                            G_output_out, x_batch = sess.run([G_output_sample, x], feed_dict={random:random_noise, y:y_batch, mask:block_mask, inverse_mask:inverse_block_mask})

                            #visualize
                            G_output_out = G_output_out*inverse_block_mask + y_batch*block_mask
                            G_output_out = np.squeeze(G_output_out)
                            plot(x_batch, name=str(step)+'-noiseX' ,output_path=self.output_path)
                            plot(y_batch, name=str(step)+'-realX' ,output_path=self.output_path)
                            plot(G_output_out, name=str(step)+'-fakeG' ,output_path=self.output_path)
                            
                            #sample form testing data
                            _, idx_train_test, y_batch = self.train_test_dataloader.load_batch(self.batch_size, idx_train_test, size=[96, 96])
                            random_noise = np.random.normal(size=y_batch.shape)
                            G_output_out, x_batch = sess.run([G_output_sample, x], feed_dict={random:random_noise, y:y_batch, mask:block_mask, inverse_mask:inverse_block_mask})

                            #visualize
                            G_output_out = G_output_out*inverse_block_mask + y_batch*block_mask
                            G_output_out = np.squeeze(G_output_out)
                            plot(x_batch, name=str(step)+'-noiseX-train' ,output_path=self.output_path)
                            plot(y_batch, name=str(step)+'-realX-train' ,output_path=self.output_path)
                            plot(G_output_out, name=str(step)+'-fakeG-train' ,output_path=self.output_path)
                        
                        #save model    
                        if step % 1000 == 0:
                            print('Saving model...')
                            saver.save(sess, self.model_path + 'model', global_step=step)
                        
                        #finish training    
                        if epoch == self.epoch:
                            print('Finish training...')
                            break

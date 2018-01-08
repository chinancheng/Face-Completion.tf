import tensorflow as tf

class Model:
    
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def generator(self, x, dropout=False, reuse=False):
        with tf.variable_scope('G') as scope:
            if reuse:
                scope.reuse_variables()
            feat = self.downsample(x)
            feat = tf.layers.conv2d_transpose(feat, 1024, (1, 1), padding='same', strides=(3, 3), kernel_initializer=tf.contrib.layers.xavier_initializer())
            img = self.build_up_resnet(feat)
            out = tf.nn.sigmoid(img)
      
            return out
      
    def discriminator(self, x, dropout=True, reuse=False):
        with tf.variable_scope('D') as scope:
            if reuse:
                scope.reuse_variables()
            feat = self.build_down_resnet(x)
            feat = tf.layers.flatten(feat)
            if dropout:
                feat = tf.nn.dropout(feat, 0.5)
            out = tf.layers.dense(feat, 1, name='fc')
      
            return out
   
    def downsample(self, x):
        with tf.variable_scope('Downsample'):
            feat = self.build_down_resnet(x)
            feat = tf.layers.conv2d(feat, 1024, (1, 1), padding='same', strides=(1, 1), kernel_initializer=tf.contrib.layers.xavier_initializer())
       
            return feat


    def build_down_resnet(self, x):
        
        conv1 = tf.layers.conv2d(x, 64, (3, 3), padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1')       
        with tf.variable_scope('block1'):
            block1 = self.build_residual_block(conv1, 64, (2, 2))
        with tf.variable_scope('block2'):
            block2 = self.build_residual_block(block1, 128, (2, 2))
        with tf.variable_scope('block3'):
            block3 = self.build_residual_block(block2, 256, (2, 2))
        with tf.variable_scope('block4'):
            block4 = self.build_residual_block(block3, 512, (2, 2))
        with tf.variable_scope('block5'):
            block5 = self.build_residual_block(block4, 1024, (2, 2))
        feat = tf.layers.average_pooling2d(block5, (3, 3), (1, 1))
        
        return feat
   
    
    def build_up_resnet(self, feat_reshape):
        
        with tf.variable_scope('block1'):
            block1 = self.build_residual_block(feat_reshape, 1024, (2, 2), transpose=True)
        with tf.variable_scope('block2'):
            block2 = self.build_residual_block(block1, 512, (2, 2), transpose=True)
        with tf.variable_scope('block3'):
            block3 = self.build_residual_block(block2, 256, (2, 2), transpose=True)
        with tf.variable_scope('block4'):
            block4 = self.build_residual_block(block3, 128, (2, 2), transpose=True)
        with tf.variable_scope('block5'):
            block5 = self.build_residual_block(block4, 64, (2, 2), transpose=True)   
        deconv5 = tf.layers.conv2d_transpose(block5, 3, (3, 3), padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv5')
              
        return deconv5
  
    
    def build_residual_block(self, input_, channel, strides, transpose=False):
        
        if not transpose:
            bn = self.lrelu(tf.layers.batch_normalization(input_))
            conv1 = tf.layers.conv2d(bn, channel, (3, 3), padding='same', strides=strides, kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv2 = self.lrelu(tf.layers.batch_normalization(conv1))
            conv2 = tf.layers.conv2d(conv2, channel, (3, 3), padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv3 =tf.layers.conv2d(input_, channel, (1, 1), strides=strides, kernel_initializer=tf.contrib.layers.xavier_initializer())
            out = tf.add(conv3, conv2)            
        else:
            bn = tf.nn.relu(tf.layers.batch_normalization(input_))
            deconv1 = tf.layers.conv2d_transpose(bn, channel, (3, 3), padding='same', strides=strides, kernel_initializer=tf.contrib.layers.xavier_initializer())
            deconv2 = tf.nn.relu(tf.layers.batch_normalization(deconv1))
            deconv2 = tf.layers.conv2d_transpose(deconv2, channel, (3, 3), padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
            deconv3 =tf.layers.conv2d_transpose(input_, channel, (1, 1), strides=strides, kernel_initializer=tf.contrib.layers.xavier_initializer())
            out = tf.add(deconv3, deconv2)
        
        return out
    
    def lrelu(self, x, leak=0.2):
        return tf.maximum(x, leak*x)


import tensorflow as tf

def kl_loss(mean, std):
    return tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(std) -2.0 * tf.log(std + 1e-8) - 1.0))

def lrelu(x, leak=0.2):
        return tf.maximum(x, leak*x)


def batch_norm(x, reuse=False):
    return tf.layers.batch_normalization(x, epsilon=1e-5, momentum = 0.9, name='batch_norm', reuse=reuse)


def conv2d(x, name, h, w, c_in, c_out, strides=[1, 1, 1, 1], activation=True, is_batch_norm=False):
    with tf.variable_scope(name):
        w = tf.get_variable(name='w', 
                            shape=[h, w, c_in, c_out], 
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b', 
                            shape=[c_out],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        out = tf.nn.conv2d(input=x, filter=w, strides=strides, padding='SAME', name='conv')
        
        if is_batch_norm:    
            out = batch_norm(out)

        if activation:
            out = tf.nn.relu(out)
    
    return out


def deconv2d(x, name, h, w, c_in, c_out, out_h, out_w, strides, batch_size, activation=True, is_batch_norm=False):
    with tf.variable_scope(name):
        w = tf.get_variable(name='w', 
                            shape=[h, w, c_out, c_in], 
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b', 
                            shape=[c_out],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d_transpose(value=x, filter=w, 
                                      output_shape=[batch_size, out_h,  out_w, c_out], 
                                      strides=strides, padding='SAME', name='deconv')
        out = tf.nn.bias_add(conv, b)

        if is_batch_norm:    
            out = batch_norm(out)

        if activation:
            out = tf.nn.relu(out)
        
    return out

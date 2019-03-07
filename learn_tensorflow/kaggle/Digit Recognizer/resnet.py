import tensorflow as tf


def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return tf.nn.max_pool2d(inputs, [1, 1 ,1,1], stride=factor, scope=scope)


def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections = None, scope = None):
    
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = inputs.shape[3]

        preact = tf.nn.batch_normalization(inputs, scope='preact')
        preact = tf.nn.relu(preact, nanme=None)
 
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = tf.nn.conv2d(preact, [1, 1, 1 ,1], stride = stride, scope = 'shortcut')
        
        residual = tf.nn.conv2d(preact, depth_bottleneck, [1, 1,1 ,1], stride = 1, scope = 'conv1')
        residual = tf.nn.conv2d(residual, depth_bottleneck, 3, stride, scope = 'conv2')
        residual = tf.nn.conv2d(residual, depth, [1, 1], stride = 1, normalizer_fn = None, activation_fn = None, scope = 'conv3')
 
        output = shortcut + residual
 
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)

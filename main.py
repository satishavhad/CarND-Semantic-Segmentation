import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import time
import scipy
import shutil

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

#Override the helper function save_inference_samples with additional logic for saving checkpoints
#copied from helper.py and then modified
def save_inference_samples1(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, epoch):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = helper.gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
    
    # keep checkpoints at different steps while training a model. https://www.tensorflow.org/api_docs/python/tf/train/Saver
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(output_dir, 'SS_fcn_{}.ckpt'.format(epoch)))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # Use tf.saved_model.loader.load to load the model and weights 
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    # Get Tuples of Tensors from VGG model loaded in above step
    vgg16_graph = tf.get_default_graph()
    image_input = vgg16_graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob   = vgg16_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out  = vgg16_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out  = vgg16_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out  = vgg16_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # Implementing the FCN decoder. Get the logits for Layer 3,4 and 7.
    layer7_logits = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, name='layer7_logits')
    layer4_logits = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=1, name='layer4_logits')
    layer3_logits = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=1, name='layer3_logits')

    # Upsample by transposed convolution and combine the result of the layers through elementwise addition (skip connection)
    transposed_layer1 = tf.layers.conv2d_transpose(layer7_logits, num_classes, kernel_size=4, strides=(2, 2),padding='same', name='transposed_layer1')
    transposed_layer2 = tf.add(transposed_layer1, layer4_logits, name='transposed_layer2')

    # Upsample by transposed convolution and combine the result of the layers through elementwise addition (skip connection)
    transposed_layer3 = tf.layers.conv2d_transpose(transposed_layer2, num_classes, kernel_size=4, strides=(2, 2), padding='same', name='transposed_layer3')
    transposed_layer4 = tf.add(transposed_layer3, layer3_logits, name='transposed_layer4')

    # return the final layer
    return tf.layers.conv2d_transpose(transposed_layer4, num_classes, kernel_size=16, strides=(8, 8), padding='same', name='transposed_layer4')
    
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    
    # the output tensor is 4D so we have to reshape it to 2D
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # Use standard cross entropy loss function and then Adam Optimizer to optimize the network
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, runs_dir=None, data_dir=None, image_shape=None, logits=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    
    Added 4 new inputs to the function runs_dir, data_dir, image_shape and logits
    """
    # TODO: Implement function
    # Create the runs directory if does not exist before
    if runs_dir is not None:
        if not os.path.exists(runs_dir):
            os.makedirs(runs_dir)

    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        Network_loss = 0
        No_of_images = 0

        # Do Checkoint saver frequently
        if data_dir is not None:
            if i > 0 and (i % 20) == 0:
                save_inference_samples1(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, i)

        # Train neural network
        for X_train, y_train in get_batches_fn(batch_size):
            loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict={input_image: X_train, correct_label: y_train, keep_prob: 0.8})
            Network_loss += loss
            No_of_images += len(X_train)

        print("Epoch {} has training loss: {}".format(i, Network_loss/No_of_images))
 
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    
    # Hyper-parameters
    epochs = 20
    batches = 1
    learning_rate = 0.0001  

    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, tf.constant(learning_rate), num_classes)        
                
        # TODO: Train NN using the train_nn function    
        train_nn(sess, epochs, batches, get_batches_fn, train_op, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate, runs_dir, data_dir, image_shape, logits)
        
        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        save_inference_samples1(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input, 'END_EPOCH')
        
        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()

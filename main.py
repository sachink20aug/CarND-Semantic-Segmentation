#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import cv2
from PIL import Image
import glob

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def create_video():
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    w=576
    h=160
    video = cv2.VideoWriter('output.avi',fourcc, 2.0, (w,h))
    for root, dirs, files in os.walk('runs'):
        for name in files:
            if name.endswith((".png")):
                im=cv2.imread(root+'/'+name)
                video.write(im)
    video.release()      
    cv2.destroyAllWindows()
    print("Video Saved")
    
def load_vgg(sess, vgg_path):

	# Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_layer = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return (input_layer, keep_prob, layer3_out, layer4_out, layer7_out)
    
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
	
    
	#Add 1x1 convolution layer to vgg_layer7_out with stride 1 and kernel 1
	conv_1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
	#Add deconvolution layer after conv_1 layer with a 4x4 kernel and stride of 2 for upsampling
	deconv_1 = tf.layers.conv2d_transpose(conv_1, num_classes, (4, 4), (2, 2), padding = 'SAME', kernel_initializer=tf.random_normal_initializer(stddev=0.01))
	#Add 1x1 convolution layer to vgg_layer4_out with stride 1 and kernel 1
	conv_2 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
	#Add skip layer 
	skip_1 = tf.add(deconv_1,conv_2)
	#Add deconvolution layer after conv_1 layer with a 4x4 kernel and stride of 2 for upsampling to the skip layer
	deconv_2 = tf.layers.conv2d_transpose(skip_1, num_classes, (4, 4), (2, 2), padding='SAME', kernel_initializer=tf.random_normal_initializer(stddev=0.01))
	#Add 1x1 convolution layer to vgg_layer3_out with stride 1 and kernel 1
	conv_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
	#Add skip layer 
	skip_2 = tf.add(deconv_2,conv_3)
	#Add deconvolution layer to skip layer 
	deconv_3 = tf.layers.conv2d_transpose(skip_2, num_classes, (16,16), strides=(8, 8), padding='SAME', kernel_initializer=tf.random_normal_initializer(stddev=0.01))
	
	
	return deconv_3
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
 
    # Implement function
	logits = tf.reshape(nn_last_layer, (-1, num_classes))
	correct_label = tf.reshape(correct_label, (-1,num_classes))
	# Loss function.
	cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label))
    # Adams optimizer - adaptive learning rate
	optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    
	train_op = optimizer.minimize(cross_entropy_loss)

	return (logits, train_op, cross_entropy_loss) 
    
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):

    #Implement function
	sess.run(tf.global_variables_initializer())
    
	print("Training...")
    
	for i in range(epochs):
		print("EPOCH {} ...".format(i+1))
		for image, label in get_batches_fn(batch_size):
			_, loss = sess.run([train_op, cross_entropy_loss], 
								feed_dict={input_image: image, correct_label: label,keep_prob: 0.5, learning_rate: 0.0009})
			print("Loss: = {:.3f}".format(loss))
       
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
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

        # Build NN using load_vgg, layers, and optimize function
        epochs = 50
        batch_size = 5
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
		

        
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
		

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
				correct_label, keep_prob, learning_rate)	
        	 
        # TODO: Save inference data using helper.save_inference_samples
		
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        
        # OPTIONAL: Apply the trained model to a video
        create_video()

if __name__ == '__main__':
    run()

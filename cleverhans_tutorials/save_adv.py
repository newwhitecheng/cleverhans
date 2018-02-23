from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import pdb
import numpy as np
import scipy.io as sio
import scipy.signal as sig
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags

import logging
import os
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
from cleverhans_tutorials.tutorial_models import make_basic_cnn

FLAGS = flags.FLAGS


def mnist_tutorial_cw(train_start=0, train_end=60000, test_start=0,
                      test_end=10000, viz_enabled=False, nb_epochs=6,
                      batch_size=128, nb_classes=10, source_samples=1,
                      learning_rate=0.001, attack_iterations=100,
                      model_path=os.path.join("models", "mnist"),
                      targeted=True):
    """
    MNIST tutorial for Carlini and Wagner's attack
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param viz_enabled: (boolean) activate plots of adversarial examples
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param nb_classes: number of output classes
    :param source_samples: number of test inputs to attack
    :param learning_rate: learning rate for training
    :param model_path: path to the model file
    :param targeted: should we run a targeted attack? or untargeted?
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # MNIST-specific dimensions
    img_rows = 28
    img_cols = 28
    channels = 1

    # Set TF random seed to improve reproducibility
    #tf.set_random_seed(1234)

    # Create TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.log_device_placement=True
    sess = tf.Session(config=config)
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)
    
    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Define TF model graph
    model = make_basic_cnn()
    preds = model(x)
    print("Defined TensorFlow model graph.")
    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': os.path.join(*os.path.split(model_path)[:-1]),
        'filename': os.path.split(model_path)[-1]
    }

    rng = np.random.RandomState([2017, 8, 30])
    # check if we've trained before, and if we have, use that pre-trained model
    if os.path.exists(model_path + ".meta"):
        tf_model_load(sess, model_path)
    else:
        model_train(sess, x, y, preds, X_train, Y_train, args=train_params,
                    save=os.path.exists("models"), rng=rng)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
    assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy
    ###########################################################################
    # Craft adversarial examples using Carlini and Wagner's approach
    ###########################################################################
    nb_adv_per_sample = str(nb_classes - 1) if targeted else '1'
    # Instantiate a CW attack object
    cw = CarliniWagnerL2(model, back='tf', sess=sess)
    fgsm = FastGradientMethod(model, sess=sess)

    result = np.zeros((5,len(X_test)))
    strength = np.zeros((3,len(X_test)))

    adv_ys = None
    yname = "y"

    cw_params = {'binary_search_steps': 1,
                 'max_iterations': attack_iterations,
                 'learning_rate': 0.1,
                 'batch_size': source_samples,
                 'initial_const': 10}
    fgsm_eps = [0.1,0.3, 0.5]
    for j in fgsm_eps:
        fgsm_params = {'eps': j,
                   'clip_min': 0.,
                   'clip_max': 1.}  
                
        for i in range(len(X_test)):
            feed_dict = {x: X_test[i].reshape((1,28,28,1))}
            Classes0 = preds.eval(feed_dict=feed_dict,session=sess)
            Class0 = np.argmax(Classes0)
            result[0,i] = Class0
            adv_inputs = X_test[i]
            adv_inputs = adv_inputs.reshape((1,28,28,1))
            #adv = cw.generate_np(adv_inputs,**cw_params)
            
            adv = fgsm.generate_np(adv_inputs, **fgsm_params)
            pdb.set_trace()
	    feed_dict = {x: adv}
            Classes1 = preds.eval(feed_dict=feed_dict,session=sess)
            Class1 = np.argmax(Classes1)
            result[1,i] = Class1
            # Compute the average distortion introduced by the algorithm
            percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2,
                                           axis=(1, 2, 3))**.5)
            strength[0,i] = percent_perturbed

            adv2 = cw.generate_np(adv,**cw_params)
            feed_dict = {x: adv2}
            Classes2 = preds.eval(feed_dict=feed_dict,session=sess)
            Class2 = np.argmax(Classes2)
            result[2,i] = Class2
            # Compute the average distortion introduced by the algorithm
            percent_perturbed2 = np.mean(np.sum((adv2 - adv)**2,
                                           axis=(1, 2, 3))**.5)
            strength[1,i] = percent_perturbed2
            
            adv_f = sig.medfilt(adv,(1,3,3,1))
            feed_dict = {x: adv_f}
            Classes1 = preds.eval(feed_dict=feed_dict,session=sess)
            Class1 = np.argmax(Classes1)
            result[3,i] = Class1
            # Compute the average distortion introduced by the algorithm
            #percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2,
            #                               axis=(1, 2, 3))**.5)
            #strength[0,i] = percent_perturbed

            adv2_f = cw.generate_np(adv_f,**cw_params)
            feed_dict = {x: adv2_f}
            Classes2 = preds.eval(feed_dict=feed_dict,session=sess)
            Class2 = np.argmax(Classes2)
            result[4,i] = Class2
            # Compute the average distortion introduced by the algorithm
            percent_perturbed2 = np.mean(np.sum((adv2_f - adv_f)**2,
                                           axis=(1, 2, 3))**.5)
            strength[2,i] = percent_perturbed2
            if i%100 == 0:
                print(i)
       # exit()
    # Close TF session
    sess.close()
    sio.savemat('fgsm_mnist.mat',{'adv_01':adv_01,'adv_03':adv_03, 'adv_05':adv_05 'strength':strength})
    sio.savemat('restore_filter1.mat',{'result':result,'strength':strength})
    print('done!!!')
    # Finally, block & display a grid of all the adversarial examples
    
    return report


def main(argv=None):
    mnist_tutorial_cw(viz_enabled=FLAGS.viz_enabled,
                      nb_epochs=FLAGS.nb_epochs,
                      batch_size=FLAGS.batch_size,
                      nb_classes=FLAGS.nb_classes,
                      source_samples=FLAGS.source_samples,
                      learning_rate=FLAGS.learning_rate,
                      attack_iterations=FLAGS.attack_iterations,
                      model_path=FLAGS.model_path,
                      targeted=FLAGS.targeted)


if __name__ == '__main__':
    flags.DEFINE_boolean('viz_enabled', False, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')
    flags.DEFINE_integer('source_samples', 1, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_string('model_path', os.path.join("models", "mnist"),
                        'Path to save or load the model file')
    flags.DEFINE_integer('attack_iterations', 100,
                         'Number of iterations to run attack; 1000 is good')
    flags.DEFINE_boolean('targeted', False,
                         'Run the tutorial in targeted mode?')

    tf.app.run()

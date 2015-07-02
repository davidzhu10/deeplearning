"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""


import os
import sys
import time
import random

import numpy

import theano
import theano.tensor as T
import theano.printing as P
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd_resize import LogisticRegression, load_data
from mlp_resize import HiddenLayer

# Load gflags to define commandline flags
sys.path.append('../modules/')
import gflags
FLAGS = gflags.FLAGS

gflags.DEFINE_integer('num_epochs', 10, 'Number of epochs to run')
gflags.DEFINE_integer('batch_size', 10, 'mini batch size of images')
gflags.DEFINE_enum('model_to_run', 'conv', ['conv', 'mlp', 'lr'],
                     'Which model to run')

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

def train_models(n_epochs, n_train_batches, n_valid_batches, n_test_batches,
	           train_model, validate_model, test_model):
    print n_epochs, n_train_batches, n_valid_batches, n_test_batches

    # early-stopping parameters
    patience = 1000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)
            print "cost_ij", cost_ij

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                print "validating: "
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                print "validation losses: ", validation_losses
                for i in xrange(n_valid_batches):
                  print validation_losses[i]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
            # end if 

            print "patience = ", patience, "iter = ", iter
            if patience <= iter:
                done_looping = True
                break
        # end for
    # end while
    return (best_validation_loss, best_iter, test_score)

def build_models(learning_rate, datasets, nkerns, batch_size):
    IMG_SIZE = 400
    L0_FSIZE = 10  # filter size (width and height)
    L0_PSIZE = 5 # pool size (with and height)
    L1_FSIZE = 10
    L1_PSIZE = 5
    NUM_HIDDEN_UNITS = 10
    NUM_CATEGORIES = 2 # For digits, this is 10.

    rng = numpy.random.RandomState(23455)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,
    # IMG_SIZE * IMG_SIZE)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (IMG_SIZE, IMG_SIZE) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, IMG_SIZE, IMG_SIZE))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to 
    #   (IMG_SIZE-L0_FSIZE+1 , IMG_SIZE-L0_FSIZE+1)
    # maxpooling reduces this further by dividing them by LAYER_0_PSIZE
    # image size (width and height) of the output from this layer is thus:
    #    s1 = (IMG_SIZE-L0_FSIZE+1)/L0_PSIZE
    # 4D output tensor is thus of shape (batch_size, nkerns[0], s1, s1)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, IMG_SIZE, IMG_SIZE),
        filter_shape=(nkerns[0], 1, L0_FSIZE, L0_FSIZE),
        poolsize=(L0_PSIZE, L0_PSIZE)
    )

    l1_img_size = (IMG_SIZE-L0_FSIZE+1)/L0_PSIZE
    print("layer 1 image size", l1_img_size)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to
    # (l1_img_size-L1_FSIZE+1 , l1_img_size-L1_FSIZE+1)
    # maxpooling reduces this further to
    # l2_img_size = (l1_img_size-L1_FSIZE+1)/L1_PSIZE
    # 4D output tensor is thus of shape (batch_size, nkerns[1], l2_img_size,
    #  l2_img_size)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], l1_img_size, l1_img_size),
        filter_shape=(nkerns[1], nkerns[0], L1_FSIZE, L1_FSIZE),
        poolsize=(L1_PSIZE, L1_PSIZE)
    )
    l2_img_size = (l1_img_size-L1_FSIZE+1)/L1_PSIZE
    print("layer 2 image size", l2_img_size)

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)
    layer2_size = nkerns[1] * l2_img_size * l2_img_size
    if FLAGS.model_to_run == 'mlp':
      layer2_input = x
      layer2_size = IMG_SIZE * IMG_SIZE

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=layer2_size,
        n_out=NUM_HIDDEN_UNITS,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    lr_input = layer2.output
    lr_layer_size = NUM_HIDDEN_UNITS
    if (FLAGS.model_to_run == 'lr'):
      # Logistic Regression
      lr_input = x
      lr_layer_size = IMG_SIZE * IMG_SIZE
      
    layer3 = LogisticRegression(input=lr_input, n_in=lr_layer_size,
                                n_out=NUM_CATEGORIES)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params
    params = layer3.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, P.Print("params")(param_i) - learning_rate * P.Print("grad")(grad_i))
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1
    return (n_train_batches, n_valid_batches, n_test_batches,
            train_model, validate_model, test_model)


def evaluate_lenet5(learning_rate=0.5, n_epochs=FLAGS.num_epochs,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=FLAGS.batch_size):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    datasets = load_data(dataset)

    (n_train_batches, n_valid_batches, n_test_batches,
     train_model, validate_model, test_model) = \
        build_models(learning_rate, datasets, nkerns, batch_size)

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    start_time = time.clock()
    (best_validation_loss, best_iter, test_score) = \
        train_models(n_epochs, n_train_batches, n_valid_batches, n_test_batches,
                       train_model, validate_model, test_model)
    end_time = time.clock()

    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

def main(argv):
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError, e:
      print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
      sys.exit(1)
    evaluate_lenet5(n_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size)

if __name__ == '__main__':
    main(sys.argv)


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)

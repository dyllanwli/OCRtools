# coding=utf-8
"""
My version of CNN Sentence classification Model
@author: cer
@forked_from: Yoon Kim
"""
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import sys
import time
from cer_module import *


class CNN_Sen_Model(object):
    """
        Model class.
        img_h = sentence length (padded where necessary)
        img_w = word vector length (300 for word2vec)
        filter_hs = filter window sizes
        hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
        sqr_norm_lim = s^2 in the paper
        lr_decay = adadelta decay parameter
        """
    def __init__(self, conf):
        self.conf = conf
        print "model configs: ", self.conf
        # self.build_model()

    def build_model(self, U):
        """定义模型架构"""
        rng = np.random.RandomState(3435)

        # 导入一些模型参数
        self.img_h = self.conf["max_l"]+max(self.conf["filter_hs"])*2-2
        #self.img_h = self.conf['img_h']
        self.img_w = self.conf['img_w']
        filter_w = self.img_w
        feature_maps = self.conf['hidden_units'][0]
        filter_shapes = []
        pool_sizes = []
        for filter_h in self.conf['filter_hs']:
            filter_shapes.append((feature_maps, 1, filter_h, filter_w))
            pool_sizes.append((self.img_h - filter_h + 1, self.conf['img_w'] - filter_w + 1))

        self.x = T.matrix('x')
        self.y = T.ivector('y')
        # Words = theano.shared(value=U, name="Words")
        ################################网络架构：1.初始化###########################
        # 1.embedding层
        self.emb_layer = EmbeddingLayer(U)
        # 2.卷积层
        self.conv_layers = []
        for i in xrange(len(self.conf['filter_hs'])):
            filter_shape = filter_shapes[i]
            # print "filter_shape:", filter_shape
            pool_size = pool_sizes[i]
            conv_layer = LeNetConvPoolLayer(rng, image_shape=(self.conf['batch_size'], 1, self.img_h, self.conf['img_w']),
                                            filter_shape=filter_shape, poolsize=pool_size, non_linear=self.conf['conv_non_linear'])
            self.conv_layers.append(conv_layer)
        # 3.MLP(多层神经感知机，带dropout)
        self.conf['hidden_units'][0] = feature_maps * len(self.conf['filter_hs'])
        self.classifier = MLPDropout(rng, layer_sizes=self.conf['hidden_units'],
                                     activations=[eval(f_s) for f_s in self.conf['activations']],
                                     dropout_rates=self.conf['dropout_rate'])

        #################################网络架构：2.连接网络#########################
        # 1.embbeding层
        emb_output = self.emb_layer.build(self.x)
        # 2.卷积层
        layer0_input = emb_output
        layer1_inputs = []
        for i in xrange(len(self.conf['filter_hs'])):
            conv_layer = self.conv_layers[i]
            layer1_input = conv_layer.build(layer0_input).flatten(2)
            layer1_inputs.append(layer1_input)
        layer1_input = T.concatenate(layer1_inputs, 1)
        self.classifier.build(layer1_input)

        ###################提取模型参数########################################
        # define parameters of the model and update functions using adadelta
        params = self.classifier.params
        for conv_layer in self.conv_layers:
            params += conv_layer.params
        if self.conf["non_static"]:
            # if word vectors are allowed to change, add them as model parameters
            params += [emb_output.Words]

        self.cost = self.classifier.negative_log_likelihood(self.y)
        self.dropout_cost = self.classifier.dropout_negative_log_likelihood(self.y)
        self.grad_updates = sgd_updates_adadelta(params, self.dropout_cost, self.conf['lr_decay'],
                                            1e-6, self.conf['sqr_norm_lim'])

    def build_function(self, train_set, val_set, test_set_x):
        index = T.lscalar()
        train_set_x, train_set_y = shared_dataset((train_set[:, :self.img_h], train_set[:, -1]))
        val_set_x, val_set_y = shared_dataset((val_set[:, :self.img_h], val_set[:, -1]))
        self.val_model = theano.function([index], self.classifier.errors(self.y),
                                    givens={
                                        self.x: val_set_x[index * self.conf['batch_size']: (index + 1) * self.conf['batch_size']],
                                        self.y: val_set_y[index * self.conf['batch_size']: (index + 1) * self.conf['batch_size']]},
                                    allow_input_downcast=True)

        # compile theano functions to get train/val/test errors
        self.test_model = theano.function([index], self.classifier.errors(self.y),
                                     givens={
                                         self.x: train_set_x[index * self.conf['batch_size']: (index + 1) * self.conf['batch_size']],
                                         self.y: train_set_y[index * self.conf['batch_size']: (index + 1) * self.conf['batch_size']]},
                                     allow_input_downcast=True)
        self.train_model = theano.function([index], self.cost, updates=self.grad_updates,
                                      givens={
                                          self.x: train_set_x[index * self.conf['batch_size']:(index + 1) * self.conf['batch_size']],
                                          self.y: train_set_y[index * self.conf['batch_size']:(index + 1) * self.conf['batch_size']]},
                                      allow_input_downcast=True)
        test_pred_layers = []
        test_size = test_set_x.shape[0]
        test_layer0_input = self.emb_layer.build(test_set_x)
        print "emb_output shape : " + str(test_layer0_input.shape.eval())
        for conv_layer in self.conv_layers:
            test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
            print "conv_layer shape : " + str(test_layer0_output.shape.eval())
            test_pred_layers.append(test_layer0_output.flatten(2))
        test_layer1_input = T.concatenate(test_pred_layers, 1)
        test_y_pred = self.classifier.predict(test_layer1_input)
        test_error = T.mean(T.neq(test_y_pred, self.y))
        self.test_model_all = theano.function([self.y], test_error, allow_input_downcast=True)

    def train(self, datasets):
        # shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate
        # extra data (at random)
        np.random.seed(3435)

        if datasets[0].shape[0] % self.conf['batch_size'] > 0:
            extra_data_num = self.conf['batch_size'] - datasets[0].shape[0] % self.conf['batch_size']
            train_set = np.random.permutation(datasets[0])
            extra_data = train_set[:extra_data_num]
            new_data = np.append(datasets[0], extra_data, axis=0)
        else:
            new_data = datasets[0]
        new_data = np.random.permutation(new_data)
        n_batches = new_data.shape[0] / self.conf['batch_size']
        n_train_batches = int(np.round(n_batches * 0.9))
        # divide train set into train/val sets
        test_set_x = datasets[1][:, :self.img_h]
        test_set_y = np.asarray(datasets[1][:, -1], "int32")
        train_set = new_data[:n_train_batches * self.conf['batch_size'], :]
        val_set = new_data[n_train_batches * self.conf['batch_size']:, :]
        n_val_batches = n_batches - n_train_batches
        # build model functions
        self.build_function(train_set, val_set, test_set_x)

        # start training over mini-batches
        print '... training'
        epoch = 0
        best_val_perf = 0
        val_perf = 0
        test_perf = 0
        cost_epoch = 0
        while (epoch < self.conf['n_epochs']):
            start_time = time.time()
            epoch = epoch + 1
            if self.conf['shuffle_batch']:
                for minibatch_index in np.random.permutation(range(n_train_batches)):
                    cost_epoch = self.train_model(minibatch_index)
                    self.emb_layer.non_set_zero()
            else:
                for minibatch_index in xrange(n_train_batches):
                    cost_epoch = self.train_model(minibatch_index)
                    self.emb_layer.non_set_zero()
            train_losses = [self.test_model(i) for i in xrange(n_train_batches)]
            train_perf = 1 - np.mean(train_losses)
            val_losses = [self.val_model(i) for i in xrange(n_val_batches)]
            val_perf = 1 - np.mean(val_losses)
            print('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%'
                  % (epoch, time.time() - start_time, train_perf * 100., val_perf * 100.))
            if val_perf >= best_val_perf:
                best_val_perf = val_perf
                test_loss = self.test_model_all(test_set_y)
                test_perf = 1 - test_loss
        return test_perf



def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')


def sgd_updates_adadelta(params, cost, rho=0.95, epsilon=1e-6, norm_lim=9, word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name != 'Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates


def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to


# def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
#     """
#     Transforms sentence into a list of indices. Pad with zeroes.
#     """
#     x = []
#     pad = filter_h - 1
#     for i in xrange(pad):
#         x.append(0)
#     words = sent.split()
#     for word in words:
#         if word in word_idx_map:
#             x.append(word_idx_map[word])
#     while len(x) < max_l + 2 * pad:
#         x.append(0)
#     return x
#
#
# def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
#     """
#     Transforms sentences into a 2-d matrix.
#     """
#     train, test = [], []
#     for rev in revs:
#         sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
#         sent.append(rev["y"])
#         if rev["split"] == cv:
#             test.append(sent)
#         else:
#             train.append(sent)
#     train = np.array(train, dtype="int")
#     test = np.array(test, dtype="int")
#     return [train, test]


import sys
from time import time
import numpy as np
import theano
from theano import config
import theano.tensor as tensor

from six.moves import cPickle as pkl
from models import init_params, build_model
from optimizers import optims
from utils import np_floatX, get_minibatches_idx, load_params, init_tparams, unzip, zipp

# Set the random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)

def pred_probs(f_pred_prob, prepare_data, data, iterator):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = np.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  np.array(data[1])[valid_index])
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)

    return probs

def pred_perf(f_pred, prepare_data, data, iterator):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    for b, valid_index in iterator:
        x ,mask, y = prepare_data([data[0][t] for t in valid_index],
                                  np.array(data[1])[valid_index])
        preds = f_pred(x, mask)
        #targets = np.array(data[1])[valid_index]
        #valid_err += (preds == targets).sum()
        valid_err += (preds == y).sum()
    
    valid_err = 1. - np_floatX(valid_err) / len(data[0])

    return valid_err * 100


def _prepare_data(seqs , labels , max_l=None, pad=None):
    
    if not len(seqs):
        return None, None, None
    
    lengths = [len(s) for s in seqs]
    if max_l:
        n_seqs = []
        n_labels = []
        n_lens = []
        for l, s, y in zip(lengths, seqs, labels):
            if l <= max_l:
                n_seqs.append(s)
                n_labels.append(y)
                n_lens.append(l)
        lengths = n_lens
        labels = n_labels
        seqs = n_seqs
    
    if pad:
        max_l = max_l + 2 * pad
        x = np.zeros((max_l, len(seqs))).astype('int64')
        x_mask = np.zeros((max_l, len(seqs))).astype(theano.config.floatX)
        for idx, s in enumerate(seqs):
            x[pad:pad + lengths[idx], idx] = s
            x_mask[pad:pad + lengths[idx], idx] = 1.
    else:
        x = np.zeros((max_l, len(seqs))).astype('int64')
        x_mask = np.zeros((max_l, len(seqs))).astype(theano.config.floatX)
        for idx, s in enumerate(seqs):
            x[:lengths[idx], idx] = s
            x_mask[:lengths[idx], idx] = 1.

    labels = np.asarray(labels).astype(int)
    return x, x_mask, labels


def _load_data(revs, word_idx_map, fold=1, sort_by_len=True, valid_portion=0.1):
    train_x, train_y = [], []
    valid_x, valid_y = [], []
    test_x, test_y = [], []

    for rev in revs:
        sent = [word_idx_map[w] for w in rev['text'].split() if w in word_idx_map]

        if rev['split'] == fold:
            test_x.append(sent)
            test_y.append(rev['y'])
        else:
            train_x.append(sent)
            train_y.append(rev['y'])
    
    n_samples = len(train_x)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    
    valid_x = [train_x[i] for i in sidx[n_train:]]
    valid_y = [train_y[i] for i in sidx[n_train:]]

    train_x = [train_x[i] for i in sidx[:n_train]]
    train_y = [train_y[i] for i in sidx[:n_train]]     

    print("train: {0} - valid: {1} - test: {2}".format(len(train_x), len(valid_x), len(test_x)))
    def len_sort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_idx = len_sort(test_x)
        test_x = [test_x[i] for i in sorted_idx]
        test_set_y = [test_y[i] for i in sorted_idx]

        sorted_idx = len_sort(valid_x)
        valid_x = [valid_x[i] for i in sorted_idx]
        valid_y = [valid_y[i] for i in sorted_idx]
         
        sorted_idx = len_sort(train_x)
        train_x = [train_x[i] for i in sorted_idx]
        train_y = [train_y[i] for i in sorted_idx]

    train = train_x, train_y
    valid = valid_x, valid_y
    test = test_x, test_y

    return train, valid, test


def train_model(
    revs, word_idx_map, max_l, pad,
    valid_portion=0.1,
    n_words=10000,
    dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
    max_epochs=100,  # The maximum number of epoch to run
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.01,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer='adadelta', 
    encoder='lstm', 
    rnnshare=True,
    bidir=False,
    batch_size=16,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    W = None, # embeddings
    deep = 0, # number of layers above
    rnnlayer = 0, # number of rnn layers
    filter_hs = [3,4,5], #filter's width
    feature_maps = 100,  #number of filters
    pool_type = 'max',    #pooling type
    combine = False,
    init='uniform',
    salstm=False,
    noise_std=0.,
    dropout_penul=0.5, 
    reload_model=None,  # Path to a saved model we want to start from.
    fname='',
):
    # Model options
    optimizer = optims[optimizer]
    model_options = locals().copy()

    prepare_data = lambda x, y: _prepare_data(x, y, max_l=max_l, pad=None)
    load_data = lambda: _load_data(revs, word_idx_map, valid_portion=valid_portion)

    train, valid, test = load_data()

    ydim = np.max(train[1]) + 1
    model_options['ydim'] = ydim
    max_l = max_l + 2 * pad
    model_options['max_l'] = max_l
    print("max_l: {0} - pad: {1}".format(max_l, pad))
     
    print('building model...')
    # This create the initial parameters as np ndarrays.
    # Dict name (string) -> np ndarray

    params = init_params(model_options)

    if reload_model:
        load_params(reload_model, params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask, y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options, SEED)

    if decay_c > 0.:
        decay_c = theano.shared(np_floatX(decay_c), name='decay_c')
        weight_decay = 0.

        weight_decay += (tparams['U'] ** 2).sum()

        if model_options['encoder'] == 'lstm':
            for layer in range(model_options['deep']):
                weight_decay += (tparams['U'+str(layer+1)] ** 2).sum()
        elif model_options['encoder'] == 'cnnlstm':
            for filter_h in model_options['filter_hs']:
                weight_decay += (tparams['cnn_f'+str(filter_h)] ** 2).sum()
        
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y], cost, name='f_cost')
    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads, x, mask, y, cost)
    
    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)
    
    history_errs = []
    history_times = [time()]
    print("start training...")
    try:
        for eidx in range(max_epochs):

            # Get new shuffled index for the training set.
            kf_train = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf_train:

                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                x = [train[0][t]for t in train_index]
                y = [train[1][t] for t in train_index]
                # Get the data in np.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                cost = f_grad_shared(x, mask, y)
                f_update(lrate)

                if np.isnan(cost) or np.isinf(cost):
                    print('NaN detected')
                    return 1., 1., 1.

            use_noise.set_value(0.)

            train_err = pred_perf(f_pred, prepare_data, train, kf_train)
            valid_err = pred_perf(f_pred, prepare_data, valid, kf_valid)
            test_err = pred_perf(f_pred, prepare_data, test, kf_test)
            
            history_errs.append((train_err, valid_err, test_err))
            history_times.append(time())

            print('epoch: {0} - training time: {1} - train err: {2} - valid err: {3} - test err: {4}'.format(eidx, history_times[-1] - history_times[-2], train_err, valid_err, test_err))

    except KeyboardInterrupt:
        print("Training interupted")
        return history_errs

    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (history_times[-1] - history_times[0]) / (1. * (eidx + 1))))
    print('Training took %.1fs' % (history_times[-1] - history_times[0]))
    return history_errs

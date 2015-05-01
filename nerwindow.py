from numpy import *
from nn.base import NNBase
from nn.math import softmax, make_onehot
from misc import random_weight_matrix
import pdb


##
# Evaluation code; do not change this
##
from sklearn import metrics
def full_report(y_true, y_pred, tagnames):
    cr = metrics.classification_report(y_true, y_pred,
                                       target_names=tagnames)
    print cr

def eval_performance(y_true, y_pred, tagnames):
    pre, rec, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred)
    print "=== Performance (omitting 'O' class) ==="
    print "Mean precision:  %.02f%%" % (100*sum(pre[1:] * support[1:])/sum(support[1:]))
    print "Mean recall:     %.02f%%" % (100*sum(rec[1:] * support[1:])/sum(support[1:]))
    print "Mean F1:         %.02f%%" % (100*sum(f1[1:] * support[1:])/sum(support[1:]))


##
# Implement this!
##
class WindowMLP(NNBase):
    """Single hidden layer, plus representation learning."""

    def __init__(self, wv, windowsize=3,
                 dims=[None, 100, 5],
                 reg=0.001, alpha=0.01, rseed=10):
        """
        Initialize classifier model.

        Arguments:
        wv : initial word vectors (array |V| x D)
            note that this is the transpose of the n x |V| matrix L
            described in the handout; you'll want to keep it in
            this |V| x D form for efficiency reasons, since numpy
            stores matrix rows continguously.
        windowsize : int, size of context window
        dims : dimensions of [input, hidden, output]
            input dimension can be computed from wv.shape
        reg : regularization strength (lambda)
        alpha : default learning rate
        rseed : random initialization seed
        """

        # Set regularization
        self.lreg = float(reg)
        self.alpha = alpha # default training rate

        dims[0] = windowsize * wv.shape[1] # input dimension
        param_dims = dict(W=(dims[1], dims[0]),
                          b1=(dims[1],),
                          U=(dims[2], dims[1]),
                          b2=(dims[2],),
                          )
        param_dims_sparse = dict(L=wv.shape)

        # initialize parameters: don't change this line
        NNBase.__init__(self, param_dims, param_dims_sparse)

        random.seed(rseed) # be sure to seed this for repeatability!
        #### YOUR CODE HERE ####

        # any other initialization you need
        self.D = wv.shape[1]
        self.sparams.L = wv.copy() # store own representations
        self.params.W = random_weight_matrix(*self.params.W.shape)
        self.params.b1 = squeeze(random_weight_matrix(self.params.b1.shape[0], 1))
        self.params.U = random_weight_matrix(*self.params.U.shape)
        self.params.b2 = squeeze(random_weight_matrix(self.params.b2.shape[0], 1))

        #### END YOUR CODE ####



    def _forward(self, window):
        ##
        # Forward propagation
        # x: dim0 x 1
        # W: dim1 x dim0
        # b1: dim1 x 1
        # U: dim2 x dim1
        # b2: dim2 x 1

        x = self.sparams.L[window, :].flatten()
        z1 = self.params.W.dot(x) + self.params.b1 # dim1 x 1
        h = tanh(z1) # dim1 x1
        z2 = self.params.U.dot(h) + self.params.b2 # dim2 x 1
        y = softmax(z2) # dim2 x 1
        return x, z1, h, z2, y

    def _acc_grads(self, window, label):
        """
        Accumulate gradients, given a training point
        (window, label) of the format

        window = [x_{i-1} x_{i} x_{i+1}] # three ints
        label = {0,1,2,3,4} # single int, gives class

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.U += (your gradient dJ/dU)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # this adds an update for that index
        """
        #### YOUR CODE HERE ####

        ##
        # Forward propagation
        # x: dim0 x 1
        # W: dim1 x dim0
        # b1: dim1 x 1
        # U: dim2 x dim1
        # b2: dim2 x 1

        x, z1, h, z2, y = self._forward(window)
        label_vector = make_onehot(label, len(y))

        ##
        # Backpropagation

        dz2 = y - label_vector # dim2 x 1
        self.grads.U += outer(dz2, h)
        self.grads.U += self.lreg * self.params.U
        self.grads.b2 += dz2
        dz1 = self.params.U.T.dot(dz2) * (1 - h*h) # dim1 x 1
        self.grads.W += outer(dz1, x) # dim1 x dim0
        self.grads.W += self.lreg * self.params.W
        self.grads.b1 += dz1
        dx = self.params.W.T.dot(dz1)
        for i, w in enumerate(window):
            self.sgrads.L[w] = dx[i*self.D:(i+1)*self.D]


        #### END YOUR CODE ####


    def predict_proba(self, windows):
        """
        Predict class probabilities.

        Should return a matrix P of probabilities,
        with each row corresponding to a row of X.

        windows = array (n x windowsize),
            each row is a window of indices
        """
        # handle singleton input by making sure we have
        # a list-of-lists
        if not hasattr(windows[0], "__iter__"):
            windows = [windows]

        #### YOUR CODE HERE ####
        P = []
        for window in windows:
            _, _, _, _, y = self._forward(window)
            P.append(y)

        P = array(P)

        #### END YOUR CODE ####

        return P # rows are output for each input


    def predict(self, windows):
        """
        Predict most likely class.
        Returns a list of predicted class indices;
        input is same as to predict_proba
        """

        #### YOUR CODE HERE ####
        P = self.predict_proba(windows)
        c = argmax(P, 1)


        #### END YOUR CODE ####
        return c # list of predicted classes


    def compute_loss(self, windows, labels):
        """
        Compute the loss for a given dataset.
        windows = same as for predict_proba
        labels = list of class labels, for each row of windows
        """

        #### YOUR CODE HERE ####

        P = self.predict_proba(windows)
        J = -sum(log(choose(labels, P.T)))
        Jreg = (self.lreg / 2.0) * sum(self.params.W**2.0)
        Jreg += (self.lreg / 2.0) * sum(self.params.U**2.0)


        #### END YOUR CODE ####
        return J + Jreg

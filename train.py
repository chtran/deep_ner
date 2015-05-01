import data_utils.utils as du
import data_utils.ner as ner
from nerwindow import WindowMLP
import pdb
import numpy as np
from numpy import *

from matplotlib.pyplot import *
from nerwindow import full_report, eval_performance
import pickle


def epoch_sched(nepoch, size):
    for i in xrange(nepoch):
        for j in xrange(size):
            yield j

def random_sched(N, size):
    return np.random.randint(0, size, N)

def random_mini(k, N, size):
    for i in xrange(N/k):
        yield np.random.randint(0, size, k)

def grad_check(clf, X_train, y_train):
    clf.grad_check(X_train[0], y_train[0]) # gradient check on single point

def plot_learning_curve(clf, costs):
    counts, costs = zip(*costs)
    figure(figsize=(6,4))
    plot(5*array(counts), costs, color='b', marker='o', linestyle='-')
    title(r"Learning Curve ($\alpha$=%g, $\lambda$=%g)" % (clf.alpha, clf.lreg))
    xlabel("SGD Iterations"); ylabel(r"Average $J(\theta)$"); 
    ylim(ymin=0, ymax=max(1.1*max(costs),3*min(costs)));
    ylim(0,0.5)

    # Don't change this filename!
    savefig("ner.learningcurve.best.png")
def main():
    # Load the starter word vectors
    wv, word_to_num, num_to_word = ner.load_wv('data/ner/vocab.txt',
                                               'data/ner/wordVectors.txt')
    tagnames = ["O", "LOC", "MISC", "ORG", "PER"]
    num_to_tag = dict(enumerate(tagnames))
    tag_to_num = du.invert_dict(num_to_tag)

    # Set window size
    windowsize = 3

    # Load the training set
    docs = du.load_dataset('data/ner/train')
    X_train, y_train = du.docs_to_windows(docs, word_to_num, tag_to_num,
                                          wsize=windowsize)

    # Load the dev set (for tuning hyperparameters)
    docs = du.load_dataset('data/ner/dev')
    X_dev, y_dev = du.docs_to_windows(docs, word_to_num, tag_to_num,
                                      wsize=windowsize)

    # Load the test set (dummy labels only)
    docs = du.load_dataset('data/ner/test.masked')
    X_test, y_test = du.docs_to_windows(docs, word_to_num, tag_to_num,
                                        wsize=windowsize)
    clf = WindowMLP(wv, windowsize=windowsize, dims=[None, 100, 5],

                    reg=0.001, alpha=0.01)
    train_size = X_train.shape[0]
    """
    costs = pickle.load(open("costs.dat", "rb"))
    clf = pickle.load(open("clf.dat", "rb"))
    """
    nepoch = 5
    N = nepoch * len(y_train)
    k = 5 # minibatch size
    costs = clf.train_sgd(
            X_train, y_train, 
            idxiter=random_mini(k, N, train_size), 
            printevery=10000, costevery=10000)

    pickle.dump(clf, open("clf.dat","wb"))
    pickle.dump(costs, open("costs.dat","wb"))
    plot_learning_curve(clf, costs)
    # Predict labels on the dev set
    yp = clf.predict(X_dev)
    # Save predictions to a file, one per line
    ner.save_predictions(yp, "dev.predicted")
    full_report(y_dev, yp, tagnames) # full report, helpful diagnostics
    eval_performance(y_dev, yp, tagnames) # performance: optimize this F1
    # L: V x 50
    # W[:,50:100]: 100 x 50 
    responses = clf.sparams.L.dot(clf.params.W[:,50:100].T) # V x 100
    index = np.argsort(responses, axis=0)[::-1]

    neurons = [1,3,4,6,8] # change this to your chosen neurons
    for i in neurons:
        print "Neuron %d" % i
        top_words = [num_to_word[k] for k in index[:10,i]]
        top_scores = [responses[k,i] for k in index[:10,i]]
        print_scores(top_scores, top_words)

    
# Recommended function to print scores
# scores = list of float
# words = list of str
def print_scores(scores, words):
    for i in range(len(scores)):
        print "[%d]: (%.03f) %s" % (i, scores[i], words[i])

if __name__ == "__main__":
    main()

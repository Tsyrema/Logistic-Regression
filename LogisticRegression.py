#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split

# acquire data, split it into training and testing sets (50% each)
# nc -- number of classes for synthetic datasets
def acquire_data(data_name, nc = 2):
    if data_name == 'synthetic-easy':
        print 'Creating easy synthetic labeled dataset'
        X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes = nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 0 * rng.uniform(size=X.shape)
    elif data_name == 'synthetic-medium':
        print 'Creating medium synthetic labeled dataset'
        X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes = nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 3 * rng.uniform(size=X.shape)
    elif data_name == 'synthetic-hard':
        print 'Creating hard easy synthetic labeled dataset'
        X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes = nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 5 * rng.uniform(size=X.shape)
    elif data_name == 'moons':
        print 'Creating two moons dataset'
        X, y = datasets.make_moons(noise=0.2, random_state=0)
    elif data_name == 'circles':
        print 'Creating two circles dataset'
        X, y = datasets.make_circles(noise=0.2, factor=0.5, random_state=1)
    elif data_name == 'iris':
        print 'Loading iris dataset'
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
    elif data_name == 'digits':
        print 'Loading digits dataset'
        digits = datasets.load_digits()
        X = digits.data
        y = digits.target
    elif data_name == 'breast_cancer':
        print 'Loading breast cancer dataset'
        bcancer = datasets.load_breast_cancer()
        X = bcancer.data
        y = bcancer.target
    else:
        print 'Cannot find the requested data_name'
        assert False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    return X_train, X_test, y_train, y_test

# compare the prediction with grount-truth, evaluate the score
# y_predict and y_test
def myscore(y, y_gt):
    assert len(y) ==  len(y_gt)
    return np.sum(y == y_gt)/float(len(y))

# plot data on 2D plane
# use it for debugging
def draw_data(X_train, X_test, y_train, y_test, nclasses):

    h = .02
    X = np.vstack([X_train, X_test])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    cm = plt.cm.jet
    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm, edgecolors='k', label='Training Data')
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, edgecolors='k', marker='x', linewidth = 3, label='Test Data')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.legend()
    plt.show()

####################################################
# binary label classification

#sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


#loss function
def loss_binary(X, y, w):

    yn = sigmoid(np.dot(X, w))
    loss = (-y) * np.log(yn) - (1 - y) * np.log(1 - yn)

    return np.mean(loss)


# gradient function
def calc_grad (X, y, w):

    predictions = sigmoid(np.dot(X, w))
    gradient = np.zeros(X.shape)
    y = y.reshape((y.size,1))
    gradient = np.dot(np.transpose(X), predictions - y)/ y.size

    # print 'Xtrans.shape: ', np.transpose(X).shape
    # print 'y : ' , y.shape
    # print 'pred : ', predictions.shape
    #
    # print ' Pred - y: ', (predictions-y).shape
    # print 'Grad shape: ', gradient.shape

    return gradient


# train the weight vector w
def mytrain_binary(X_train, y_train):
    print 'Start training ...'

    threshold = 0.0001
    stepsize = 6    #initial stepsize
    w = np.zeros((X_train.shape[1], 1)) # initial w

    w_train = w     # training vector w

    loss_current = loss_binary(X_train, y_train, w)
    loss_history = np.array([])
    loss_history = np.append(loss_history, loss_current)

    iteration = 0   # to count number of iterations in the loop

    loss_difference = 100
    performance = np.array([])

    while (loss_current !=0 and loss_difference > threshold and stepsize != 0):

        start_time = time.time()

        prev_loss = loss_current
        prev_w = w_train
        prev_stepsize = stepsize

        loss_current = loss_binary(X_train,y_train, w_train)
        w_train = w_train - stepsize * calc_grad(X_train, y_train, w_train)

        new_loss = loss_binary(X_train, y_train, w_train)

        loss_difference = abs(loss_current - new_loss)

        print 'Iteration #: ' , iteration
        print 'Current loss:' , loss_current
        print 'New loss: ', new_loss
        print 'Diff in loss: ', loss_difference
        print 'Current stepsize: ', stepsize

        #
        # if (loss_current >= new_loss):
        #     iteration += 1
        #     stepsize *= 1.01
        # elif loss_current < new_loss:
        #     stepsize *= 0.5
        # loss_current = new_loss
        # loss_history = np.append(loss_history, loss_current)
        # w_train = w_train - (stepsize * calc_grad(X_train, y_train, w_train))
        # new_time = time.time()
        # total_time = new_time - start_time
        # performance = np.append(performance, total_time)


        if (loss_current >= new_loss):
            iteration +=1
            stepsize *= 1.01
            loss_current = new_loss
            loss_history = np.append(loss_history, loss_current)
            w_train = w_train - (stepsize * calc_grad(X_train, y_train, w_train))
            new_time = time.time()
            total_time = new_time - start_time
            performance = np.append(performance, total_time)
        elif loss_current < new_loss:
            # iteration +=1
            stepsize = prev_stepsize
            stepsize *= 0.5
            w_train = prev_w
            loss_current = prev_loss
            # loss_history = np.append(loss_history, loss_current)
            new_time = time.time()
            total_time = new_time - start_time
            performance = np.append(performance, total_time)

    print 'Total Iterations: ', iteration
    print 'Time of iterations: ', performance
    print ' Average time: ', np.mean(performance)
    print 'Loss values: ', loss_history
    print 'Finished training.'
    return w_train
    # return w, loss_history, iteration, performance

# compute y on any input set X, and current weights w
# returns a list of length N, each entry is between 0 and 1
def mypredict_binary(X, w):

    z = np.dot(X, w)
    y_pred = sigmoid(z)

    # return convert_pred2gt_binary(y_pred)

    return y_pred

# # convert -1/1 to 0/1
# def convert_pred2gt_binary(y_pred):
#     return np.maximum(np.zeros(y_pred.shape),y_pred)
#
# # convert 0/1 to -1/1
# def convert_gt2pred_binary(y_gt):
#     y = 2 * (y_gt-0.5)
#     return y


# predict labels using the logistic regression model on any input set X, using current w
# returns either 0 or 1
def mytest_binary(X, w):

    # assert len(w) == X.shape[1] + 1
    # w_vec = np.reshape(w, (-1, 1))
    # X_extended = np.hstack([X, np.ones([X.shape[0], 1])])
    # z = np.dot(X_extended, w_vec)
    # y_pred = sigmoid(z)

    z = np.dot(X, w)
    y_pred = sigmoid(z)

    return [1 if x >= 0.5 else 0 for x in y_pred]

    # here is a fake implementation, you should replace it
    # assert len(w) == X.shape[1] + 1
    # w_vec = np.reshape(w,(-1,1))
    # X_extended = np.hstack([X, np.ones([X.shape[0],1])])
    # y_pred = np.ravel(np.sign(np.dot(X_extended,w_vec)))
    # y_pred_final = np.maximum(np.zeros(y_pred.shape), y_pred)
    # return y_pred_final

# draw results on 2D plan for binary classification
# use it for debugging
def draw_result_binary(X_train, X_test, y_train, y_test, w):

    h = .02
    X = np.vstack([X_train, X_test])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#    ax = plt.figure(1)
    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k', label='Training Data')
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', marker='x', linewidth = 3, label='Test Data')

    # Put the result into a color plot
    tmpX = np.c_[xx.ravel(), yy.ravel()]
    Z = mypredict_binary(tmpX, w)
    Z = Z.reshape(xx.shape)

    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.RdBu, alpha = .4)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.legend()

    y_predict = mytest_binary(X_test,w)
    score = myscore(y_predict, y_test)
    plt.text(xx.max() - .3, yy.min() + .3, ('Score = %.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
    plt.show()

####################
# multi-label classification

def sigmoid_multi(z):
    z = z - np.max(z)
    sigm = np.transpose(np.exp(z)) / np.transpose(np.sum(np.exp(z), axis=1))
    print 'sigm size: ', sigm.shape
    return sigm

def loss_multi(X, y, w):
    m = X.shape[0]
    print 'rows in X: ', m

    # y_one = convert(y)

    probability = sigmoid_multi(np.dot(X,w))
    loss = - np.mean(np.sum(y * np.log(probability) + (1-y) * np.log(1- probability),axis=1))

    # loss = -1/m * np.sum(y_one * np.log(probability)) + (lam/2) * np.sum(w * w)
    return loss

def grad_multi (X, y, w):
    m = X.shape[0]
    # y_one = convert(y)
    probability = sigmoid_multi(np.dot(X,w))
    grad = -1/m * np.dot(np.transpose(X), y - probability)
    print 'Gradient size: ', grad.shape

    return grad


def convert (y):
    y_one = np.arrange(np.max(y) +1) == y[:, None].astype(float)
    print 'y_one: ', y_one
    return y_one


# train the weight vector w
def mytrain_multi(X_train, y_train):
    print 'Start training ...'

    threshold = 0.0001
    stepsize = 1  # initial stepsize
    w = np.zeros((X_train.shape[1], y_train.shape))  # initial w
    w_train = w



    # # fake code, only return a random vector
    # np.random.seed(100)
    # nfeatures = X_train.shape[1]
    # w = np.random.rand(nfeatures + 1)
    # # w = [-1,0,0]

    print 'Finished training.'
    return w_train

# compute y, NxK matrix, values between 0 and 1
def mypredict_multi(X, w):

    pred = np.argmax(sigmoid_multi(w, X),axis=1)
    return pred
    # return np.zeros([X.shape[0],1])


# predict labels using the logistic regression model on any input set X
# return length N list, value between 0 and K-1
def mytest_multi(X, w):

    pred = np.argmax(sigmoid_multi(w, X),axis=1)
    return [1 if x >= 0.5 else 0 for x in pred]
    # return np.zeros([X.shape[0],1])

################

def main():

    #######################
    # get data
    # binary labeled

    # X_train, X_test, y_train, y_test = acquire_data('synthetic-easy')
    # X_train, X_test, y_train, y_test = acquire_data('synthetic-medium')
    X_train, X_test, y_train, y_test = acquire_data('synthetic-hard')
    # X_train, X_test, y_train, y_test = acquire_data('moons')
    # X_train, X_test, y_train, y_test = acquire_data('circles')
    # X_train, X_test, y_train, y_test = acquire_data('breast_cancer')

    # multi-labeled
    # X_train, X_test, y_train, y_test = acquire_data('synthetic-easy', nc = 3)
    # X_train, X_test, y_train, y_test = acquire_data('synthetic-medium', nc = 3)
    # X_train, X_test, y_train, y_test = acquire_data('synthetic-hard', nc = 3)
    # X_train, X_test, y_train, y_test = acquire_data('iris')
    # X_train, X_test, y_train, y_test = acquire_data('digits')


    nfeatures = X_train.shape[1]    # number of features
    ntrain = X_train.shape[0]   # number of training data
    ntest = X_test.shape[0]     # number of test data
    y = np.append(y_train, y_test)
    nclasses = len(np.unique(y)) # number of classes

    # only draw data (on the first two dimension)
    draw_data(X_train, X_test, y_train, y_test, nclasses)

    if nclasses == 2:
        w_opt = mytrain_binary(X_train, y_train)
        # debugging example
        draw_result_binary(X_train, X_test, y_train, y_test, w_opt)
    else:
        w_opt = mytrain_multi(X_train, y_train)

    if nclasses == 2:
        y_train_pred = mytest_binary(X_train, w_opt)
        y_test_pred = mytest_binary(X_test, w_opt)
    else:
        y_train_pred = mytest_multi(X_train, w_opt)
        y_test_pred = mytest_multi(X_test, w_opt)

    train_score = myscore(y_train_pred, y_train)
    test_score = myscore(y_test_pred, y_test)

    print 'Training Score:', train_score
    print 'Test Score:', test_score

if __name__ == "__main__": main()
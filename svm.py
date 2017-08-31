#! /usr/bin/env python
#! -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
# from sklearn.cross_validation import train_test_split

tuned_parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
    {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.0001]},
    {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
    ]

def train_test_split(X, y, test_size=0.1, random_state=0):
    """ train_test_split

    dependency:
        numpy

    args:
        X: input parameters
        y: class value
        test size: test size ratio(0 to 1.0)
        random_state: random seed parameter

    return
        X_train: Train data(inputs)
        Y_train: Train data(results)
        X_test: Test data(inputs)
        y_test: Test data(results)
    """

    n_sample = len(X)
    np.random.seed(random_state)
    order = np.random.permutation(n_sample)
    X = X[order]
    y = y[order].astype(np.float)

    X_train = X[int(test_size * n_sample):]
    y_train = y[int(test_size * n_sample):]
    X_test = X[:int(test_size * n_sample)]
    y_test = y[:int(test_size * n_sample)]

    return X_train, X_test, y_train, y_test


def svm_tuning(X, y, score='f1', test_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_weighted' % score)
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)

    return clf

def plot_graph(X, y, param, test_size=0.1, filename="tmp"):
    """ plot_graph function

    args:
        X: input parameters
        y: class value
        param: hyper parameter dictionary

    return
        none

    output:
        visualization graph:
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    if 'gamma' in param:
        clf = svm.SVC(kernel=param['kernel'], gamma=param['gamma'], C=param['C'])
    else:
        clf = svm.SVC(kernel=param['kernel'], C=param['C'])

    clf.fit(X_train, y_train)

    plt.scatter(X[:, 0], X[:, 1], c=y , zorder=10, cmap=plt.cm.coolwarm)
    # Circle out the test data
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)

    # calculate accuracy
    # print("accuracy={0:.3f}".format(accuracy_score(y_test, clf.predict(X_test))))
    # print(accuracy_score(y_test, clf.predict(X_test)))

    # Visualization setting
    plt.axis('tight')
    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.coolwarm)
    plt.contourf(XX, YY, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    plt.title(filename)
    # plt.show()
    plt.savefig(filename)

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    # print(type(X[1]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # for 2 class
    # X = X[y != 0, :2]
    # y = y[y != 0]

    score = 'f1'
    # print(X)
    # print(y)
    clf = svm_tuning(X, y, score)

    print("# Tuning hyper-parameters for %s" % score)
    print()
    print("Best parameters set found on development set: %s" % clf.best_params_)
    print()

    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    plot_graph(X,y,clf.best_params_,filename='best_tuned')

    for i, kernel in enumerate(('linear', 'rbf', 'poly')):
        clf = svm.SVC(kernel=kernel, gamma=10, C=1)
        clf.fit(X_train, y_train)
        print("param:" + str(i))
        print(clf.get_params())

        plot_graph(X,y,clf.get_params(),filename=str(i))


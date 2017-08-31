import numpy as np
import svm
import sys

if __name__ == '__main__':
    input_file =sys.argv[1]

    table = np.loadtxt(input_file)

    X = table[:, 2:4]
    y = table[:, 0]
    y = y.astype(np.int8)
    print(X)
    print(y)

    clf = svm.svm_tuning(X, y)
    print(clf.best_params_)
    svm.plot_graph(X,y,clf.best_params_, filename="best_tuned")

    X_train, X_test, y_train, y_test = svm.train_test_split(X, y, test_size=0.1)

    for i, kernel in enumerate(('linear', 'rbf', 'poly')):
        clf = svm.SVC(kernel=kernel, gamma=10, C=1)
        clf.fit(X_train, y_train)
        print("param:" + str(i))
        print(clf.get_params())

        svm.plot_graph(X,y,clf.get_params(),filename=str(i))

#!/usr/bin/python

"""
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors

    Sara has label 0
    Chris has label 1

"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn import svm
from sklearn.metrics import accuracy_score

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### uncomment below to resize the training dataset down to 1%
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

kernel_name = "rbf"

# for c_val in [1., 10., 100., 1000., 10000.]:
for c_val in [10000.]:
    print
    print "--->", kernel_name, "  C:", c_val

    clf = svm.SVC(kernel=kernel_name, C=c_val)

    t0 = time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"

    t1 = time()
    labels_pred = clf.predict(features_test)
    print "prediction time:", round(time()-t1, 3), "s"

    acc = accuracy_score(labels_test, labels_pred)

    print "classifier accuracy:", acc
    print

    # given that Chris is 1, simpe add the values
    print sum(labels_pred)
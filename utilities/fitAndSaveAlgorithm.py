from __future__ import division
from __future__ import print_function
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
from utilities.visualize import visualize
import matplotlib.pyplot as plt
import scikitplot as skplt
import os
from joblib import dump
import time

def fitAndSaveAlgorithmPyod(detectorName, path, clf, x_train, y_train, x_test, y_test, clf_name):
    start = time.time()
    if not os.path.exists(path):
        os.makedirs(path)

    clf.fit(x_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(x_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(x_test)  # outlier scores

    skplt.metrics.plot_confusion_matrix(
    y_train, y_train_pred, normalize=False, title="Confusion Matrix on Train Set with " + detectorName)
    plt.savefig('{path_images}confusionMatrixTrain_{clf_name}.png'.format(clf_name=clf_name, path_images=path))
    plt.show()

    skplt.metrics.plot_confusion_matrix(
    y_test, y_test_pred, normalize=False, title="Confusion Matrix on Test Set with " + detectorName)
    plt.savefig('{path_images}confusionMatrixTest_{clf_name}.png'.format(clf_name=clf_name, path_images=path))
    plt.show()

    if x_train.shape[1] == 2: # visualize the results
        visualize(clf_name, x_train, y_train, x_test, y_test, y_train_pred,
                  y_test_pred, show_figure=True, save_figure=True, path_save_figure=path)

    # save the model
    try:
        dump(clf, path + clf_name + '.joblib')
    except Exception as e:
        print("No se ha podido guardar " + clf_name)

    end = time.time()
    duration = end-start
    print("duration: " + str(duration))

    fileDuration = open(path + "duration_" + clf_name + ".txt", "w")
    fileDuration.write(str(duration))
    fileDuration.close()

def fitAndSaveAlgorithmSupervised(detectorName, path, clf, x_train, y_train, x_test, y_test, clf_name):
    start = time.time()
    if not os.path.exists(path):
        os.makedirs(path)

    clf.fit(x_train, y_train)

    # get the prediction on the test data
    y_test_pred = clf.predict(x_test)  # outlier labels (0 or 1)

    skplt.metrics.plot_confusion_matrix(
    y_test, y_test_pred, normalize=False, title="Confusion Matrix on Test Set with " + detectorName)
    plt.savefig('{path_images}confusionMatrixTest_{clf_name}.png'.format(clf_name=clf_name, path_images=path))
    plt.show()

    # save the model
    try:
        dump(clf, path + clf_name + '.joblib')
    except Exception as e:
        print("No se ha podido guardar " + clf_name)

    end = time.time()
    duration = end-start
    print("duration: " + str(duration))

    fileDuration = open(path + "duration_" + clf_name + ".txt", "w")
    fileDuration.write(str(duration))
    fileDuration.close()

def fitAndSaveAlgorithmUnsupervised(detectorName, path, clf, x_train, x_test, y_test, clf_name):
    start = time.time()
    if not os.path.exists(path):
        os.makedirs(path)

    clf.fit(x_train)

    # get the prediction on the test data
    y_test_pred = clf.predict(x_test)

    skplt.metrics.plot_confusion_matrix(
    y_test, y_test_pred, normalize=False, title="Confusion Matrix on Test Set with " + detectorName)
    plt.savefig('{path_images}confusionMatrixTest_{clf_name}.png'.format(clf_name=clf_name, path_images=path))
    plt.show()

    # save the model
    try:
        dump(clf, path + clf_name + '.joblib')
    except Exception as e:
        print("No se ha podido guardar " + clf_name)

    end = time.time()
    duration = end-start
    print("duration: " + str(duration))

    fileDuration = open(path + "duration_" + clf_name + ".txt", "w")
    fileDuration.write(str(duration))
    fileDuration.close()

from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
from pyod.utils.data import check_consistent_shape
from pyod.utils.data import get_outliers_inliers
import scikitplot as skplt
from sklearn.metrics import average_precision_score, plot_precision_recall_curve


def visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
              y_test_pred, show_figure=True,
              save_figure=False, path_save_figure='images/'):

    def _add_sub_plot(X_inliers, X_outliers, sub_plot_title,
                      inlier_color='blue', outlier_color='orange'):
        plt.axis("equal")
        plt.scatter(X_inliers[:, 0], X_inliers[:, 1], label='inliers',
                    color=inlier_color, s=40)
        plt.scatter(X_outliers[:, 0], X_outliers[:, 1],
                    label='outliers', color=outlier_color, s=50, marker='^')
        plt.title(sub_plot_title, fontsize=15)
        plt.xticks([])
        plt.yticks([])
        plt.legend(loc=3, prop={'size': 10})

    # check input data shapes are consistent
    X_train, y_train, X_test, y_test, y_train_pred, y_test_pred = \
        check_consistent_shape(X_train, y_train, X_test, y_test, y_train_pred,
                               y_test_pred)

    if X_train.shape[1] != 2:
        raise ValueError("Input data has to be 2-d for visualization. The "
                         "input data has {shape}.".format(shape=X_train.shape))

    X_train_outliers, X_train_inliers = get_outliers_inliers(X_train, y_train)
    X_train_outliers_pred, X_train_inliers_pred = get_outliers_inliers(
        X_train, y_train_pred)

    X_test_outliers, X_test_inliers = get_outliers_inliers(X_test, y_test)
    X_test_outliers_pred, X_test_inliers_pred = get_outliers_inliers(
        X_test, y_test_pred)

    # plot ground truth vs. predicted results
    fig = plt.figure(figsize=(12, 10))
    plt.suptitle("{clf_name} Detector".format(clf_name=clf_name),
                 fontsize=15)

    fig.add_subplot(221)
    _add_sub_plot(X_train_inliers, X_train_outliers, 'Train Set Ground Truth',
                  inlier_color='blue', outlier_color='orange')

    fig.add_subplot(222)
    _add_sub_plot(X_train_inliers_pred, X_train_outliers_pred,
                  'Train Set Prediction', inlier_color='blue',
                  outlier_color='orange')

    fig.add_subplot(223)
    _add_sub_plot(X_test_inliers, X_test_outliers, 'Test Set Ground Truth',
                  inlier_color='green', outlier_color='red')

    fig.add_subplot(224)
    _add_sub_plot(X_test_inliers_pred, X_test_outliers_pred,
                  'Test Set Prediction', inlier_color='green',
                  outlier_color='red')

    if save_figure:
        plt.savefig('{path_images}{clf_name}.png'.format(clf_name=clf_name, path_images=path_save_figure), dpi=300)

    if show_figure:
        plt.show()

def visualizeTest(clf_name, x_test, y_test,
                  y_test_pred, show_figure=True,
                  save_figure=False, path_save_figure='images/'):

    def _add_sub_plot(X_inliers, X_outliers, sub_plot_title,
                      inlier_color='blue', outlier_color='orange'):
        plt.axis("equal")
        plt.scatter(X_inliers[:, 0], X_inliers[:, 1], label='inliers',
                    color=inlier_color, s=40)
        plt.scatter(X_outliers[:, 0], X_outliers[:, 1],
                    label='outliers', color=outlier_color, s=50, marker='^')
        plt.title(sub_plot_title, fontsize=15)
        plt.xticks([])
        plt.yticks([])
        plt.legend(loc=3, prop={'size': 10})

    if x_test.shape[1] != 2:
        raise ValueError("Input data has to be 2-d for visualization. The "
                         "input data has {shape}.".format(shape=x_test.shape))

    X_test_outliers, X_test_inliers = get_outliers_inliers(x_test, y_test)
    X_test_outliers_pred, X_test_inliers_pred = get_outliers_inliers(
        x_test, y_test_pred)

    # plot ground truth vs. predicted results
    fig = plt.figure(figsize=(12, 10))
    plt.suptitle("{clf_name} Detector".format(clf_name=clf_name),
                 fontsize=15)

    fig.add_subplot(121)
    _add_sub_plot(X_test_inliers, X_test_outliers, 'Test Set Ground Truth',
                  inlier_color='green', outlier_color='red')

    fig.add_subplot(122)
    _add_sub_plot(X_test_inliers_pred, X_test_outliers_pred,
                  'Test Set Prediction', inlier_color='green',
                  outlier_color='red')

    if save_figure:
        plt.savefig('{path_images}2D_plot_{clf_name}.png'.format(clf_name=clf_name, path_images=path_save_figure), dpi=300)

    if show_figure:
        plt.show()

def visualize_confusion_matrix(y_test, y_score, clf_name, path):
    skplt.metrics.plot_confusion_matrix(
    y_test, y_score, normalize=False, title="Confusion Matrix with " + clf_name)
    plt.savefig('{path_images}confusionMatrixTest_{clf_name}.png'.format(clf_name=clf_name, path_images=path))
    plt.show()

def visualize_precision_recall(classifier, x_test, y_test, y_score, clf_name, path):
    average_precision = average_precision_score(y_test, y_score)
    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))
    disp = plot_precision_recall_curve(classifier, x_test, y_test)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                       'AP={0:0.2f}'.format(average_precision))
    plt.savefig('{path_images}PrecisionRecallCurve_{clf_name}.png'.format(clf_name=clf_name, path_images=path))
    plt.show()
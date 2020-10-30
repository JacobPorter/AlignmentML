import datetime
import itertools
import optparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix'):
    # cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.style.use('ggplot')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest')  # , cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(list(range(cm.shape[0])),
                                  list(range(cm.shape[1]))):
        plt.text(j,
                 i,
                 format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix


def plot(y_test, y_pred, class_names):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=4)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix,
                          classes=class_names,
                          title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix,
                          classes=class_names,
                          normalize=True,
                          title='Normalized confusion matrix')
    plt.show()


def main():
    now = datetime.datetime.now()
    usage = "usage: %prog [options] <feature_file> <labels_file> "
    description = ""
    p = optparse.OptionParser(usage=usage, description=description)
    sys.stdout.write("The AlignmentTypeML was started at %s\n" % (str(now)))
    p.add_option(
        '--n_jobs',
        '-n',
        help=
        'The number of jobs (parallel processes) to use to do machine learning. [default: %default]',
        default='4')
    p.add_option(
        '--datasize',
        '-s',
        help=
        'The number of rows of the data to do machine learning. [default: %default]',
        default=None)
    p.add_option(
        '--testsize',
        '-t',
        help=
        'The number of rows of the data to calculate test set error. [default: %default]',
        default=100000)
    p.add_option(
        '--classifier',
        '-c',
        help=
        'Specify the classifier to use.  Choices are "RF", "GB", "LR", "MLP", "SVC", "Ensemble", "Random".  [default: %default]',
        default='RF')
    p.add_option(
        '--write_data',
        '-w',
        help=
        'Write the training and the test data to a file with prefix given in this option. [default: %default]',
        default=None)
    p.add_option('--bayesOpt',
                 '-b',
                 help='Use Bayesian optimization.',
                 action='store_true',
                 default=False)
    p.add_option(
        '--acqFunc',
        '-a',
        help=
        'Choose the aquisition function for Bayesian optimization.  The bayesOpt parameter must be True. Choices are "LCB", "EI", "PI", "gp_hedge".  [default: %default]',
        default="gp_hedge")
    p.add_option(
        '--numIter',
        '-i',
        help=
        'The number of iterations (samples) for Bayesian optimization.  [default: %default]',
        default=25)
    options, args = p.parse_args()
    if len(args) == 0:
        p.print_help()
        return
    if len(args) != 2:
        p.error("There must be two files given in the arguments.")
    if not os.path.exists(args[0]) or not os.path.exists(args[1]):
        p.error("One of the files in the arguments could not be found.")
    sys.stdout.write(
        "Executing AlignmentTypeML on feature file {0} and label file {1}.  There will be {2} processes.  The training data size is {3}, and the test size is {4}.\n"
        .format(args[0], args[1], options.n_jobs, options.datasize,
                options.testsize))
    sys.stdout.write(
        "Classifier: {0}\nBayesian Optimization: {1}, {2}, {3}.\n".format(
            options.classifier, options.bayesOpt, options.acqFunc,
            options.numIter))
    sys.stdout.flush()
    doML(getX(args[0]), getY(args[1]), int(options.n_jobs), options.classifier,
         int(options.datasize), int(options.testsize), options.write_data,
         options.bayesOpt, options.acqFunc, int(options.numIter))
    later = datetime.datetime.now()
    sys.stdout.write(
        "The AlignmentTypeML was started at %s and took %s time.\n" %
        (str(now), str(later - now)))


if __name__ == "__main__":
    main()

import random
import numpy as np
import matplotlib.pyplot as plt

from get_data import get_data
from models import DecisionTree, train_error, entropy, gini_index


def loss_plot(ax, title, tree, pruned_tree, train_data, test_data):
    '''
        Example plotting code. This plots four curves: the training and testing
        average loss using tree and pruned tree.
        You do not need to change this code!
        Arguments:
            - ax: A matplotlib Axes instance.
            - title: A title for the graph (string)
            - tree: An unpruned DecisionTree instance
            - pruned_tree: A pruned DecisionTree instance
            - train_data: Training dataset returned from get_data
            - test_data: Test dataset returned from get_data
    '''
    fontsize=8
    ax.plot(tree.loss_plot_vec(train_data), label='train non-pruned')
    ax.plot(tree.loss_plot_vec(test_data), label='test non-pruned')
    ax.plot(pruned_tree.loss_plot_vec(train_data), label='train pruned')
    ax.plot(pruned_tree.loss_plot_vec(test_data), label='test pruned')


    ax.locator_params(nbins=3)
    ax.set_xlabel('number of nodes', fontsize=fontsize)
    ax.set_ylabel('loss', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    legend = ax.legend(loc='upper center', shadow=True, fontsize=fontsize-2)

def explore_dataset(filename, class_name):
    train_data, validation_data, test_data = get_data(filename, class_name)

    regular_train_error = DecisionTree (data=train_data, gain_function=train_error, validation_data=None)
    pruned_train_error = DecisionTree (data=train_data, gain_function=train_error, validation_data=validation_data)
    regular_entropy = DecisionTree (data=train_data, gain_function=entropy, validation_data=None)
    pruned_entropy = DecisionTree (data=train_data, gain_function=entropy, validation_data=validation_data)
    regular_gini = DecisionTree (data=train_data, gain_function=gini_index, validation_data=None)
    pruned_gini = DecisionTree (data=train_data, gain_function=gini_index, validation_data=validation_data)

    # TODO: Print 12 loss values associated with the dataset.
    # For each measure of gain (training error, entropy, gini):
    #      (a) Print average training loss (not-pruned)
    print ('Non-Pruned Training Error Training Loss', regular_train_error.loss(train_data))
    print ('Non-Pruned Entropy Training Loss', regular_entropy.loss(train_data))
    print ('Non-Pruned Gini Training Loss', regular_gini.loss(train_data))
         # (b) Print average test loss (not-pruned)
    print ('Non-Pruned Training Error Test Loss', regular_train_error.loss(test_data))
    print ('Non-Pruned Entropy Test Loss', regular_entropy.loss(test_data))
    print ('Non-Pruned Gini Test Loss', regular_gini.loss(test_data))

    #      (c) Print average training loss (pruned)
    print ('Pruned Training Error Training Loss', pruned_train_error.loss(train_data))
    print ('Pruned Entropy Training Loss', pruned_entropy.loss(train_data))
    print ('Pruned Gini Training Loss', pruned_gini.loss(train_data))
    #      (d) Print average test loss (pruned)
    print ('Pruned Training Error Test Loss', pruned_train_error.loss(test_data))
    print ('Pruned Entropy Test Loss', pruned_entropy.loss(test_data))
    print ('Pruned Gini Test Loss', pruned_gini.loss(test_data))


    # TODO: Feel free to print or plot anything you like here. Just comment
    # make sure to comment it out, or put it in a function that isn't called
    # by default when you hand in your code!

def main():
    ########### PLEASE DO NOT CHANGE THESE LINES OF CODE! ###################
    random.seed(1)
    np.random.seed(1)
    #########################################################################

    explore_dataset('data/chess.csv', 'won')
    explore_dataset('data/spam.csv', '1')

main()

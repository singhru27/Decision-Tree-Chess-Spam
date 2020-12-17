import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt


def train_error(prob):
    '''
        TODO:
        Calculate the train error of the subdataset and return it.
        For a dataset with two classes, C(p) = min{p, 1-p}. DONE
    '''

    percentage_true = prob
    percentage_false = 1 - percentage_true

    train_error = min(percentage_true, percentage_false)

    return train_error



def entropy(prob):
    '''
        TODO:
        Calculate the entropy of the subdataset and return it.
        For a dataset with 2 classes, C(p) = -p * log(p) - (1-p) * log(1-p)
        For the purposes of this calculation, assume 0*log0 = 0.
    '''


    percentage_true = prob
    percentage_false = 1 - percentage_true

    # Handling the case when p = 0
    if percentage_true == 0:
        train_error = -(1 - percentage_true)*math.log(1 - percentage_true)
    elif percentage_true == 1:
        train_error = (-percentage_true)*math.log(percentage_true)
    else:
        train_error = (-percentage_true)*math.log(percentage_true) - (1 - percentage_true)*math.log(1 - percentage_true)


    return train_error


def gini_index(prob):
    '''
        TODO:
        Calculate the gini index of the subdataset and return it.
        For dataset with 2 classes, C(p) = 2 * p * (1-p)
    '''

    percentage_true = prob
    percentage_false = 1 - percentage_true

    train_error = 2 * percentage_true * percentage_false
    return train_error

class Node:
    '''
    Helper to construct the tree structure.
    '''
    def __init__(self, left=None, right=None, depth=0, index_split_on=0, isleaf=False, label=1):
        self.left = left
        self.right = right
        self.depth = depth
        self.index_split_on = index_split_on
        self.isleaf = isleaf
        self.label = label
        self.info = {} # used for visualization


    def _set_info(self, gain, num_samples):
        '''
        Helper function to add to info attribute.
        You do not need to modify this.
        '''

        self.info['gain'] = gain
        self.info['num_samples'] = num_samples


class DecisionTree:

    def __init__(self, data, validation_data=None, gain_function=entropy, max_depth=40):
        self.max_depth = max_depth
        self.root = Node()
        self.gain_function = gain_function

        indices = list(range(1, len(data[0])))

        self.root.depth = 0
        self._split_recurs(self.root, data, indices)

        # Pruning
        if validation_data is not None:
            self._prune_recurs(self.root, validation_data)

        x = np.arange(69)



    def predict(self, features):
        '''
        Helper function to predict the label given a row of features.
        You do not need to modify this.
        '''
        return self._predict_recurs(self.root, features)


    def accuracy(self, data):
        '''
        Helper function to calculate the accuracy on the given data.
        You do not need to modify this.
        '''
        return 1 - self.loss(data)


    def loss(self, data):
        '''
        Helper function to calculate the loss on the given data.
        You do not need to modify this.
        '''
        cnt = 0.0
        test_Y = [row[0] for row in data]
        for i in range(len(data)):
            prediction = self.predict(data[i])
            if (prediction != test_Y[i]):
                cnt += 1.0
        return cnt/len(data)


    def _predict_recurs(self, node, row):
        '''
        Helper function to predict the label given a row of features.
        Traverse the tree until leaves to get the label.
        You do not need to modify this.
        '''
        if node.isleaf or node.index_split_on == 0:
            return node.label
        split_index = node.index_split_on
        if not row[split_index]:
            return self._predict_recurs(node.left, row)
        else:
            return self._predict_recurs(node.right, row)


    def _prune_recurs(self, node, validation_data):
        '''
        TODO:
        Prune the tree bottom up recursively. Nothing needs to be returned.
        Do not prune if the node is a leaf.
        Do not prune if the node is non-leaf and has at least one non-leaf child.
        Prune if deleting the node could reduce loss on the validation data.
        '''

        if node.isleaf == True:
            return

        self._prune_recurs(node.left, validation_data)
        self._prune_recurs(node.right, validation_data)

        if node.left.isleaf == False or node.right.isleaf == False:
            return

        current_loss = self.loss (validation_data)

        ## Testing if pruning reduces loss
        node.isleaf = True

        # If pruning does not reduce loss, we reset the node to not be a leaf
        if self.loss(validation_data) > current_loss:
            node.isleaf = False
        # If pruning does reduce loss, we set the node to be a leaf and eliminate
        # its children
        else:
            node.left = None
            node.right = None


    def _is_terminal(self, node, data, indices):
        '''
        TODO:
        Helper function to determine whether the node should stop splitting.
        Stop the recursion if:
            1. The dataset is empty.
            2. There are no more indices to split on.
            3. All the instances in this dataset belong to the same class
            4. The depth of the node reaches the maximum depth.
        Return:
            - A boolean, True indicating the current node should be a leaf.
            - A label, indicating the label of the leaf (or the label it would
              be if we were to terminate at that node)
        '''


        ## Handling the case for which the dataset is empty
        if data.size == 0:
            return [True, 1]

        ## This is used to determine the number of examples classified as 1,
        ## and the number of examples classified as 0, to be returned if this is a _is_terminal
        num_examples = np.shape(data)[0]
        num_true = 0

        for i in range (num_examples):
            if data[i][0] == 1:
                num_true = num_true + 1

        num_false = num_examples - num_true

        ## Handling the case for which there are no more indices to split on
        if len(indices) == 0:
            if num_true > num_false:
                return [True, 1]
            else:
                return [True, 0]

        ## Handling the case for which all the instances in the data set belong to the same class
        if num_true == 0:
            return [True, 0]
        if num_false == 0:
            return [True, 1]

        ## Handling the case for which the maximum depth has been reached
        if node.depth == self.max_depth:
            if num_true > num_false:
                return [True, 1]
            else:
                return [True, 0]

        ## Handling the normal case, in which the node is not a terminal node
        if num_true > num_false:
            return [False, 1]
        else:
            return [False, 0]


    def _split_recurs(self, node, data, indices):
        '''
        TODO:
        Recursively split the node based on the rows and indices given.
        Nothing needs to be returned.

        First use _is_terminal() to check if the node needs to be split.
        If so, select the column that has the maximum infomation gain to split on.
        Store the label predicted for this node, the split column, and use _set_info()
        to keep track of the gain and the number of datapoints at the split.
        Then, split the data based on its value in the selected column.
        The data should be recursively passed to the children.
        '''

        ## This handles the case if the node is a terminal node. If it is, the label
        ## of the node is set, it is designated as a leaf, and the function returns
        terminal_list = self._is_terminal(node, data, indices)

        if terminal_list[0] == True:
            node.label = terminal_list[1]
            node.isleaf = True
            return

        ## If the selected node is not a terminal node, we need to find the split
        ## index which provides the greatest information gain.
        maximum_gain = 0
        split_index = None


        for index in indices:
            if self._calc_gain(data, index, self.gain_function) > maximum_gain:
                maximum_gain = self._calc_gain(data, index, self.gain_function)
                split_index = index

        ## If none of the indices are increasing the gain, we just randomly select an index
        ## to split on
        if split_index == None:
            split_index = indices[0]

        ## Storing the predicted label for this node
        node.label = terminal_list[1]
        node.index_split_on = split_index
        node._set_info(maximum_gain, np.shape(data)[0])

        ## Crating datasets that house the data which are true and false respectively
        ## for the desired split index
        true_data = data[data[:,split_index] == True]
        false_data = data[data[:,split_index] == False]

        ## Creating a left node and a right nodes
        new_left_node = Node(depth=node.depth + 1)
        new_right_node = Node(depth=node.depth + 1)

        ## Setting the children of the parent node to the new_left_node and the
        ## new_right_node
        node.left = new_left_node
        node.right = new_right_node

        ## Creating a new copy of the list of indices and removing the already split index
        new_indices = copy.copy(indices)
        new_indices.remove(split_index)

        ## Recursively calling this function to continue building the trees
        self._split_recurs(new_left_node, false_data, new_indices)
        self._split_recurs(new_right_node, true_data, new_indices)



    def _calc_gain(self, data, split_index, gain_function):
        '''
        TODO:
        Calculate the gain of the proposed splitting and return it.
        Gain = C(P[y=1]) - (P[x_i=True] * C(P[y=1|x_i=True]) + (P[x_i=False] * C(P[y=1|x_i=False]))
        Here the C(p) is the gain_function. For example, if C(p) = min(p, 1-p), this would be
        considering training error gain. Other alternatives are entropy and gini functions.
        '''

        ## Calculating P[x_i=True] and P[x_i=False]
        num_examples = np.shape(data)[0]
        count_true = 0

        for i in range (num_examples):
            if data[i][split_index] == True:
                count_true = count_true + 1

        probability_true = count_true/num_examples
        probability_false = 1 - probability_true

        ## Calculating error with no split
        count_label_1 = 0
        for i in range (num_examples):
            if data[i][0] == True:
                count_label_1 = count_label_1 + 1

        probability_label1_overall = count_label_1 / num_examples

        # Calculating the error on the unsplit dataset
        original_error = gain_function(probability_label1_overall)


        ## Crating datasets with only true or false values for the given split_index
        true_data = data[(data[:,split_index] == True)]
        false_data = data[(data[:,split_index] == False)]

        ## Calculating the percentage of "1" labels in the true_dataset and the false_dataset based on the split
        # Positive split percentage
        count_label_1 = 0
        num_examples_true_data = np.shape(true_data)[0]

        for i in range (num_examples_true_data):
            if true_data[i][0] == 1:
                count_label_1 = count_label_1 + 1

        # Handling the edge case in which this dataset does not have any examples
        if num_examples_true_data == 0:
            percentage_label_1_true_data = 0
        else:
            percentage_label_1_true_data = count_label_1 / num_examples_true_data

        # Negative split percentage
        count_label_1 = 0
        num_examples_false_data = np.shape(false_data)[0]

        for i in range(num_examples_false_data):
            if false_data[i][0] == 1:
                count_label_1 = count_label_1 + 1

        # Handling the edge case in which this sub dataset does not have any examples
        if num_examples_false_data == 0:
            percentage_label_1_false_data = 0
        else:
            percentage_label_1_false_data = count_label_1 / num_examples_false_data

        gain = original_error - probability_true*gain_function(percentage_label_1_true_data) - probability_false*gain_function(percentage_label_1_false_data)

        return gain


    def print_tree(self):
        '''
        Helper function for tree_visualization.
        Only effective with very shallow trees.
        You do not need to modify this.
        '''
        print('---START PRINT TREE---')
        def print_subtree(node, indent=''):
            if node is None:
                return str("None")
            if node.isleaf:
                return str(node.label)
            else:
                decision = 'split attribute = {:d}; gain = {:f}; number of samples = {:d}'.format(node.index_split_on, node.info['gain'], node.info['num_samples'])
            left = indent + '0 -> '+ print_subtree(node.left, indent + '\t\t')
            right = indent + '1 -> '+ print_subtree(node.right, indent + '\t\t')
            return (decision + '\n' + left + '\n' + right)

        print(print_subtree(self.root))
        print('----END PRINT TREE---')


    def loss_plot_vec(self, data):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        self._loss_plot_recurs(self.root, data, 0)
        loss_vec = []
        q = [self.root]
        num_correct = 0
        while len(q) > 0:
            node = q.pop(0)
            num_correct = num_correct + node.info['curr_num_correct']
            loss_vec.append(num_correct)
            if node.left != None:
                q.append(node.left)
            if node.right != None:
                q.append(node.right)

        return 1 - np.array(loss_vec)/len(data)


    def _loss_plot_recurs(self, node, rows, prev_num_correct):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        labels = [row[0] for row in rows]
        curr_num_correct = labels.count(node.label) - prev_num_correct
        node.info['curr_num_correct'] = curr_num_correct

        if not node.isleaf:
            left_data, right_data = [], []
            left_num_correct, right_num_correct = 0, 0
            for row in rows:
                if not row[node.index_split_on]:
                    left_data.append(row)
                else:
                    right_data.append(row)

            left_labels = [row[0] for row in left_data]
            left_num_correct = left_labels.count(node.label)
            right_labels = [row[0] for row in right_data]
            right_num_correct = right_labels.count(node.label)

            if node.left != None:
                self._loss_plot_recurs(node.left, left_data, left_num_correct)
            if node.right != None:
                self._loss_plot_recurs(node.right, right_data, right_num_correct)

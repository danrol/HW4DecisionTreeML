import numpy as np
import pandas as pd
import glob
import graphviz as gv
import sys
import coloredlogs
import logging
import os

coloredlogs.install()
log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log.setLevel(logging.INFO)

'''
Suggest three different improvements to the algorithm that could improve
the result (add your answer to the theoretical part).
'''

train_data = []
test_data = []
header_attributes = []
features = []


def load_data():
    global train_data, test_data, features
    log.info("entered load_data")
    features = ['checking_status', 'saving_status', 'credit_history', 'housing', 'job', 'property_magnitude',
                'number_of_dependents', 'number_of_existing_credits', 'own_telephone', 'foreign_workers', 'label']
    train_data = pd.read_csv("dataset/train.txt", header=None, names=features)
    test_data = pd.read_csv("dataset/val.txt", header=None, names=features)

    features = ['checking_status', 'saving_status', 'credit_history', 'housing', 'job', 'property_magnitude',
                'number_of_dependents', 'number_of_existing_credits', 'own_telephone', 'foreign_workers', 'label']
    # header_attributes = {features[0]: ['x', 'n', 'b', 'g'], features[1]: ['n', 'b', 'm', 'g', 'w'],
    #                      features[2]: ['a', 'c', 'd', 'e', 'n'], features[3]}
    get_attributes('dataset/header_attributes.txt')
    log.info(f'\nheader_attributes: {header_attributes}')
    log.info('finished to load data')



def get_attributes(filename):
    global header_attributes
    file = open(filename).read()
    file = file.split('\n')
    header_attributes = [line.split(',') for line in file]


def get_enthropy(values, labels):
    for value, label in zip(values, labels):
        pass

def get_gains(dataset):
    pass


def decision_tree_build():
    '''
    Implement a method called decision tree build, which builds the decision
    tree using the training data. You are allowed to use pandas library.
    :return:
    '''

    for feature in features:
        pass


def plot_tree():
    '''
    Use the graphviz library to plot the final tree. At each note write the
    computed Entropy and the Information-Gain. You should present the
    results with 5 decimal places. Submit your result as plot.png
    :return:
    '''

    decision_tree = gv.Digraph('unix', filename='unix.gv',
            node_attr={'color': 'lightblue2', 'style': 'filled'})
    decision_tree.attr(size='6,6')
    pass

def print_accuracy():
    '''
    Implement a method called print accuracy. This method should evaluate
    and print the accuracy of your model.

    :return:
    '''
    pass


def main():
    load_data()
    print(f'train_data: {train_data}\n test_data: {test_data}')


if __name__ == "__main__":
    main()
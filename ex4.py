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
header_attributes = dict()
headers = []
label_header = 'label'
labels_column = []
classifiers = ["G", "B"]
num_of_rows = None


def load_data():
    global train_data, test_data, headers, label_header, labels_column, num_of_rows
    log.info("entered load_data")
    headers = ['checking_status', 'saving_status', 'credit_history', 'housing', 'job', 'property_magnitude',
                'number_of_dependents', 'number_of_existing_credits', 'own_telephone', 'foreign_workers', 'label']
    train_data = pd.read_csv("dataset/train.txt", header=None, names=headers)
    headers = headers[:-1]
    # test_data = pd.read_csv("dataset/val.txt", header=None, names=headers)

    # header_attributes = {features[0]: ['x', 'n', 'b', 'g'], features[1]: ['n', 'b', 'm', 'g', 'w'],
    #                      features[2]: ['a', 'c', 'd', 'e', 'n'], features[3]}
    get_attributes('dataset/header_attributes.txt')
    labels_column = train_data[label_header]
    num_of_rows = len(labels_column)
    log.info(f'number of columns = {num_of_rows}')
    train_data.drop(label_header, axis=1, inplace=True)
    # log.info(f'\nlabel_column: \n{label_column}')
    log.info(f'\nheaders: \n{train_data.columns.tolist()}')
    log.info(f'\nheader_attributes: \n{header_attributes}')
    log.info('finished to load data')



def get_attributes(filename):
    global header_attributes, headers
    file = open(filename).read()
    file = file.split('\n')
    header_attributes = {header_name: line.split(',') for header_name, line in zip(headers, file)}
    header_attributes[label_header] = ['G', 'B']


def init_header_classifiers_counter(header):
    global header_attributes
    output_counter_for_header_attributes = dict()
    for attribute in header_attributes[header]:
        output_counter_for_header_attributes[attribute] = dict()
        for classifier in classifiers:
            output_counter_for_header_attributes[attribute][classifier] = 0
    log.info(f'\nheader = {header}, output_counter_for_header_attributes: \n{output_counter_for_header_attributes}')
    return output_counter_for_header_attributes


def count_classifiers_per_attribute(header):
    output_counter_for_header_attributes = init_header_classifiers_counter(header)
    for attribute, label in zip(train_data[header], labels_column):
        attribute = str(attribute)
        label = str(label)
        if attribute in output_counter_for_header_attributes:
            if label in output_counter_for_header_attributes[attribute]:
                output_counter_for_header_attributes[attribute][label] += 1
            else:
                print(f'label = {label} is not in output_counter_for_header_attributes')
        else:
            print(f'attribute = {attribute} is not in output_counter_for_header_attributes ')
        #alternative:

    return output_counter_for_header_attributes


def get_probabilities(output_counter_for_header_attributes):
    probabilities = output_counter_for_header_attributes
    for attribute, classifiers_counts in output_counter_for_header_attributes.items():
        for classifier, count in classifiers_counts.items():
            probabilities[attribute][classifier] = count/num_of_rows
    return probabilities


def get_header_entropy(header):
    output_counter_for_header_attributes = count_classifiers_per_attribute(header)
    log.info(f'\noutput_counter_for_header_attributes in header = {header}: \n{output_counter_for_header_attributes}')
    probabilities = get_probabilities(output_counter_for_header_attributes)
    log.info(f'\nprobabilities for header = {header}:\n{probabilities}')
    header_entropy = 0
    for classifier_probabilities in probabilities.values():
        for classifier, probability in classifier_probabilities.items():
            header_entropy = -probability*np.log2(probability)
    log.info(f'header = {header}, header_entropy = {header_entropy}')


def get_gain(header):
    pass


def decision_tree_build():
    '''
    Implement a method called decision tree build, which builds the decision
    tree using the training data. You are allowed to use pandas library.
    :return:
    '''
    global headers
    log.info('entered decision_tree_build')
    for header in headers:
        get_header_entropy(header)
    # get_gain(headers[0])
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
    decision_tree_build()
    # print(f'train_data: {train_data}\n test_data: {test_data}')


if __name__ == "__main__":
    main()
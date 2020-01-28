'''!!!! Works only with short amount of input data because of stackoverflow !!!!!!'''

import numpy as np
import pandas as pd
import sys
from numpy import log2 as log
import pprint


eps = np.finfo(float).eps

sys.setrecursionlimit(100000)
headers = ['checking_status', 'saving_status', 'credit_history', 'housing', 'job', 'property_magnitude',
                'number_of_dependents', 'number_of_existing_credits', 'own_telephone', 'foreign_workers', 'label']
dataset = pd.read_csv("dataset/short_train.txt", header=None, names=headers)
label_column_name = 'label'
df = pd.DataFrame(dataset,columns=headers)

print(sys.getrecursionlimit())




def find_entropy(df):
    key = df.keys()[-1]  # To make the code generic, changing target variable class name
    entropy = 0
    values = df[key].unique()
    for value in values:
        fraction = df[key].value_counts()[value] / len(df[key])
        entropy += -fraction * np.log2(fraction)
    return entropy


def find_entropy_attribute(df, attribute):
    key = df.keys()[-1]  # To make the code generic, changing target variable class name
    target_variables = df[key].unique()  # This gives all 'Yes' and 'No'
    variables = df[
        attribute].unique()  # This gives different features in that attribute (like 'Hot','Cold' in Temperature)
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute] == variable][df[key] == target_variable])
            den = len(df[attribute][df[attribute] == variable])
            fraction = num / (den + eps)
            entropy += -fraction * log(fraction + eps)
        fraction2 = den / len(df)
        entropy2 += -fraction2 * entropy
    return abs(entropy2)


def find_winner(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
        #         Entropy_att.append(find_entropy_attribute(df,key))
        IG.append(find_entropy(df) - find_entropy_attribute(df, key))
    return df.keys()[:-1][np.argmax(IG)]


def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)


def buildTree(df, tree=None):
    key = df.keys()[-1]  # To make the code generic, changing target variable class name

    # Here we build our decision tree

    # Get attribute with maximum information gain
    node = find_winner(df)

    # Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
    attValue = np.unique(df[node])

    # Create an empty dictionary to create tree
    if tree is None:
        tree = {}
        tree[node] = {}

    # We make loop to construct a tree by calling this function recursively.
    # In this we check if the subset is pure and stops if it is pure.
    for value in attValue:
        print(f'running')
        subtable = get_subtable(df, node, value)
        clValue, counts = np.unique(subtable[label_column_name], return_counts=True)

        if len(counts) == 1:  # Checking purity of subset
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = buildTree(subtable)  # Calling the function recursively

    return tree


def predict(inst, tree):
    # This function is used to predict for any input variable

    # Recursively we go through the tree that we built earlier

    for nodes in tree.keys():

        value = inst[nodes]
        tree = tree[nodes][value]
        prediction = 0

        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break

    return prediction


def main():
    tree = buildTree(df)
    pprint.pprint(tree)
    pass

if __name__ == "__main__":
    main()
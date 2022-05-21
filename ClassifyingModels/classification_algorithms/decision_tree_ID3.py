"""
Author:
            Name                Email
        Israel Avihail      bilbisli@gmail.com
"""

import copy
import pandas as pd
from project_util.classifier_algorithm import ClassifierAlgorithm
from project_util.convert_to_list import convert_to_list
from project_util.dictionary_tree import DictionaryTree
from project_util.information_gain import information_gain


class DecisionTreeID3(ClassifierAlgorithm):
    """
    This class represents a decision tree ID3 classifier algorithm
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor method
        """
        super().__init__({'decision_tree', 'Decision_Tree', 'DecisionTree', 'Decision_tree', 'decision', 'Decision',
                          'd', 'D', 'dt', 'DT', 'd_t', 'D_T', 'd.t', 'D.T', 'Dt', 'D_t', 'D.t'})
        self.x = None
        self.y = None
        self.tree = DictionaryTree()
        self.min_gain = args[0] if len(args) >= 1 else kwargs['min_gain'] if 'min_gain' in kwargs else 0.0
        self.min_samples_leaf = args[1] if len(args) >= 2 else kwargs['min_samples_leaf'] \
            if 'min_samples_leaf' in kwargs else 0.0
        self.max_depth = args[2] if len(args) >= 3 else kwargs['max_depth'] if 'max_depth' in kwargs else 100

    def fit(self, x, y):
        # convert parameters into DataFrames
        self.x = pd.DataFrame(x)
        self.y = pd.Series(y)

        def split_tree(path, df, depth=0):
            """
            This function is responsible of splitting the current node of the decision tree (where the path is reached)
            :param path: the currently built path (branch) of the tree
            :type path: Sequence
            :param df: the current dataFrame to split
            :type df: DataFrame
            :param depth: count high of the tree
            :type depth: int
            :return: nothing
            :rtype: None
            """
            if not df.drop('class', axis=1).empty:
                # finds the best column to split the tree with (with the maximum info gain)
                best_column = best_column_gain_split(df.drop('class', axis=1), df['class'])
                classifiers_list = convert_to_list(df['class'])
                if not best_column:
                    path.append(max(set(classifiers_list), key=classifiers_list.count))
                    self.tree.add_node(path)
                else:
                    depth += 1
                    if depth >= self.max_depth:
                        path.append(max(set(classifiers_list), key=classifiers_list.count))
                        self.tree.add_node(path)
                        return
                    # adds the best column to the tree path
                    path.append(best_column)
                    # applies the updated path to the decision tree together with the column's unique values
                    # self.tree.add_node(path, self.x[best_column].unique())
                    self.tree.add_node(path, df[best_column].unique())
                    # save the majority rule of that node in case future paths are invalid thus classification needs to
                    # be according to the node
                    self.tree.add_node(path + [-1] + [max(set(classifiers_list), key=classifiers_list.count)])
                    # finds the next level of the tree - where each value leads to
                    # (next column if more splits are needed or classification if no more splits are required)
                    # next_level(path, df, best_column, self.x[best_column].unique())
                    next_level(path, df, best_column, depth, df[best_column].unique())

        def best_column_gain_split(x_=None, y_=None):
            """
            This function is responsible of finding the current best split of the decision tree
            :param x_: the attributes in array-like / DataFrame form (all columns excluding classifier column)
            :type x_: DataFrame
            :param y_: the classifier in array-like / DataFrame (column) form
            :type y_: Series
            :return: the best column (by name) that gives the highest information gain
            :rtype: str
            """
            x_ = self.x if x is None else x_
            y_ = self.y if y is None else y_
            best = 0
            column_found = None
            for column in x_:
                temp = information_gain(y_, x_[column])
                if temp >= best:
                    best = temp
                    column_found = column
            if best < self.min_gain:
                return None
            return column_found

        def next_level(path, df, father, depth=1, *values):
            """
            This method handles constructing the next level of the tree -
            a column's values are given and for each value the function finds where it leads to
            (next column if more splits are needed or classification if no more splits are required)
            :param path: the currently built path (branch) of the tree
            :type path: Sequence
            :param df: the current dataFrame used to construct the next level
            :type df: DataFrame
            :param father: the column containing the unique values sent (father node of the values nodes)
            :type father: str
            :param values: the unique values of the column (branches of the tree / children of the column node)
            :type values: tuple(str,)
            :param depth: count high of the tree
            :type depth: int
            :return: nothing
            :rtype: None
            """
            for value in values[0]:
                # path is copied so each value path (branch) will not affect the other
                value_path = copy.deepcopy(path)
                # adds the value to it's newly created path
                value_path.append(value)
                # narrows the classifiers in the dataFrame to include only the unique rows which have the value within
                # the father (column) in order to check if a single classification has been reached or further splitting
                # is needed
                classifiers = df[df[father] == value]['class']
                if 1 < classifiers.nunique() and classifiers.size > self.min_samples_leaf:
                    # if there are more than one unique classification for the value in the column - further splitting
                    # is done
                    split_tree(value_path, df[df[father] == value].drop(father, axis=1), copy.deepcopy(depth))
                else:
                    # if there is a single unique classification for the value in the column - a classification has been
                    # reached and it is made to be a leaf in the tree
                    classifiers_list = convert_to_list(classifiers)
                    value_path.append(max(set(classifiers_list), key=classifiers_list.count))
                    self.tree.add_node(value_path)

        # combines the data with the classifying column
        combined_data = pd.DataFrame(x).copy()
        combined_data['class'] = y
        # constructs the decision tree
        split_tree([], combined_data)

        return self

    def predict_one_line(self, x):
        """
        This method uses the built decision tree constructed in 'fit' in order to give a prediction for one given line
        :param x: the line to predict (classify)
        :return: the prediction (classification)
        """
        # convert parameters into DataFrames
        x = pd.DataFrame(x)
        # no predictions can be made if parameters \ data are empty
        if x.empty or self.y.empty or self.x.empty:
            return
        # the path of the decision tree to reach the desired leaf
        path = []
        # the current node of the tree
        node = list(self.tree.root.keys())[0]
        while True:
            # adds the column to the path
            path.append(node)
            # try-except for the case that no splitting has been made thus the classification will only be the majority
            # law hence the tree will only have one node which will be the most common classification
            try:
                # retrieves the value of that column
                node = x[node].iloc[0]
                # adds the value to the path
                path.append(node)
            except KeyError:
                pass
            # retrieves the next column according to the value together with indication if a leaf was reached
            values = self.tree.get_value_by_path(path)
            # for the cases that the given path is valid until the last node and the rest is invalid
            if len(values) > 2:
                return self.tree.get_value_by_path(path[:-1] + [-1])[0]
            node, is_leaf = values
            # if a leaf is reached that means it's a prediction (classification)
            if is_leaf:
                return node

    def predict(self, x):
        # convert parameters into DataFrames
        x = pd.DataFrame(x)
        # no predictions can be made if parameters \ data are empty
        if x.empty or self.y.empty or self.x.empty:
            return
        # the list of predictions (classifications\results)
        result = []
        # sends each line to 'predict_one_line' in order to predict it's classification
        for line in x.itertuples(index=False):
            # reconstructs the line as DataFrame with the proper columns
            line = pd.DataFrame({k: [v] for k, v in zip(x.columns, line)})
            result.append(self.predict_one_line(line))
        return result

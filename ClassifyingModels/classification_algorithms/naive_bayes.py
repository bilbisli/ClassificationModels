"""
Author:
            Name                Email
        Israel Avihail      bilbisli@gmail.com
"""

from project_util.classifier_algorithm import ClassifierAlgorithm
import pandas as pd


class NaiveBayes(ClassifierAlgorithm):
    """
    This class represents a naive bayesian classifier algorithm
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__({'naive_bayes', 'Naive_Bayes', 'NaiveBayes', 'Naive_bayes', 'Naive', 'naive', 'n', 'N',
                          'nb', 'NB', 'n_b', 'N_B', 'n.b', 'N.B', 'Nb', 'N_b', 'N.b'})
        self.x = None
        self.y = None
        self.value_probabilities = {}
        self.value_count = {}

    def fit(self, x, y):
        # convert parameters into DataFrames
        self.x = pd.DataFrame(x)
        self.y = pd.Series(y)
        # nothing to fir if parameters are empty
        if self.y.empty or self.x.empty:
            return
        # data probability dictionary
        self.value_probabilities['data'] = {}
        # classifiers probability dictionary
        self.value_probabilities['classifiers'] = {}
        # classifiers count dictionary
        self.value_count = self.y.value_counts().to_dict()
        # calculate classifiers probabilities
        size = len(self.y)
        for classifier in self.y.unique():
            self.value_probabilities['classifiers'][classifier] = self.y[self.y == classifier].count() / size
        # calculate data probabilities
        united_df = self.x.copy()
        united_df[self.y.name] = self.y
        for column in self.x:
            for value in self.x[column].unique():
                for classifier in self.value_count:
                    # initiate dictionaries
                    if column not in self.value_probabilities['data']:
                        self.value_probabilities['data'][column] = {}
                    if value not in self.value_probabilities['data'][column]:
                        self.value_probabilities['data'][column][value] = {}
                    # for each unique value in each column - calculates the probabilities according to each classifier
                    # applies laplacian correction
                    self.value_probabilities['data'][column][value][classifier] = \
                        (united_df[column][(united_df[column] == value) &
                                           (united_df[self.y.name] == classifier)].count() + 1) / \
                        (self.value_count[classifier] + len(self.value_count))
        return self

    def predict_one_line(self, x):
        """
        This method uses the calculated probabilities done in 'fit' in order to give a prediction for one given line
        :param x: the line to predict (classify)
        :return: the prediction (classification)
        """
        # convert parameters into DataFrames
        x = pd.DataFrame(x)
        # no predictions can be made if parameters \ data are empty
        if x.empty or self.y.empty or self.x.empty:
            return
        # dictionary for storing each classifier's probability based on the given line (x)
        calculation = {}
        for classifier in self.value_probabilities['classifiers']:
            calculation[classifier] = self.value_probabilities['classifiers'][classifier]
            for column in x:
                value = x[column].iloc[0]
                # for the case that test data has values which weren't in the train data (will be value 0)
                # if value not in self.value_probabilities['data'][column]:
                #     continue
                try:
                    calculation[classifier] *= self.value_probabilities['data'][column][value][classifier]
                except KeyError:
                    continue
        # returns the most likely (probable) classification
        return max(calculation, key=calculation.get)

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


# nb = NaiveBayes()
# print(nb.get_strings())

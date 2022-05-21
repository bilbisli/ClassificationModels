"""
Author:
            Name                Email
        Israel Avihail      bilbisli@gmail.com
"""

import abc


class ClassifierAlgorithm(abc.ABC):

    def __init__(self, strings=None):
        # string of algorithm names
        self.strings = {self.__class__.__name__} if strings is None else strings

    @abc.abstractmethod
    def fit(self, x, y):
        """
        This method fits (trains) the algorithm of the classifier
        :param x: the attributes in array-like / DataFrame form (all columns excluding classifier column)
        :param y: the classifier in array-like / DataFrame (column) form
        :return: the fitted model
        :rtype: ClassifierAlgorithm
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x):
        """
        This method predicts the classification of given data based on the fitted data
        :param x: the attributes in array-like / DataFrame form
        :return: the predictions in array-like form
        """
        raise NotImplementedError

    def get_strings(self):
        return self.strings

    def set_strings(self, strings):
        self.strings = strings

    def algorithm_by_name(self, name):
        if isinstance(name, str):
            if any(name.casefold() == nick_name.casefold() for nick_name in self.strings):
                return self

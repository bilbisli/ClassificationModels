"""
Author:
            Name                Email
        Israel Avihail      bilbisli@gmail.com
"""

from project_util.classifier_algorithm import ClassifierAlgorithm
from project_util.container import Container


class ClassifierContainer(Container, ClassifierAlgorithm):
    def __init__(self, classifier_algorithm, *extras, **kwextras):
        Container.__init__(self, *extras, **kwextras)
        ClassifierAlgorithm.__init__(self)
        self.classifier_algorithm = classifier_algorithm
        self.extras = extras
        self.kwextras = kwextras



    def fit(self, *args, **kwargs):
        return self.classifier_algorithm.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.classifier_algorithm.predict(*args, **kwargs)

    def get_classifier_algorithm(self):
        return self.classifier_algorithm

"""
Author:
            Name                Email
        Israel Avihail      bilbisli@gmail.com
"""


from sklearn.metrics import classification_report, confusion_matrix


def precision_check(y_original, y_prediction):
    print('Classification report:')
    print(classification_report(y_original, y_prediction))
    print('Confusion matrix:')
    print(confusion_matrix(y_original, y_prediction))

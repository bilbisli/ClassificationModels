"""
Author:
            Name                Email
        Israel Avihail      bilbisli@gmail.com
"""


import pandas as pd


def convert_to_list(data):
    """
    This function converts given data to list
    :param data: the given data
    :return: the given data as list
    :rtype: list
    """
    if data is None:
        return []
    if isinstance(data, pd.DataFrame):
        data = list(data.iloc[:, 0])
    elif isinstance(data, pd.Series):
        data = list(data)
        if isinstance(data[0], list):
            data = data[0]
    elif isinstance(data, str):
        data = [data]
    else:
        data = list(data)
    return data

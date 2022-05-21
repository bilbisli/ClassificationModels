"""
Author:
            Name                Email
        Israel Avihail      bilbisli@gmail.com
"""

import copy
import os
import pickle
from os.path import exists
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import classification_algorithms
from classification_algorithms import DecisionTreeID3
from preprocessing.entropy_discretization import EntropyDiscretization
from preprocessing.equal_depth_discretization import EqualDepthDiscretization
from preprocessing.equal_width_discretization import EqualWidthDiscretization
from project_util.classifier_container import ClassifierContainer
from project_util.container import Container
from project_util.convert_to_list import convert_to_list
from project_util.precision_check import precision_check


class DataAnalysis:
    def __init__(self, args):
        self.args = args
        # initialize the path variable with a valid directory
        self.path = self.check_path(args.path + '/' if 'path' in args else '')
        # initialize the file variable with a valid directory
        self.file = self.check_path(self.path + '/' + args.train, 'file') if 'train' in args else None
        self.file = self.file[self.file.rfind('/') + 1:] if 'train' in args else None
        self.data = pd.read_csv(self.path + self.file) if self.file else None
        self.save_name = args.save_name if 'save_name' in args else None
        self.test_file = self.check_path(self.path + '/' + args.test, 'file') if 'test' in args else None
        self.test_file = self.test_file[self.test_file.rfind('/') + 1:] if 'test' in args else None
        self.test_data = pd.read_csv(self.path + self.test_file) if self.test_file else None
        self.fill_type = args.fill if 'fill' in args else 'classifier'
        self.normalization = args.normalization if 'normalization' in args else False
        # scaler for train data normalization (and apply the same normalization on test data )
        self.scale = StandardScaler()
        self.discretization = args.discretization if 'discretization' in args else None
        self.bins = args.bins if 'bins' in args and args.bins else 6
        self.bins = int(self.bins[0]) if isinstance(self.bins, Sequence) and len(self.bins) == 1 else self.bins
        self.algorithm = args.algorithm if 'algorithm' in args else 'dt'
        self.implementation = args.implementation if 'implementation' in args and args.implementation == 'own' else None
        self.classifier = 'class'  # 'Major_category'
        self.model = None
        self.model_name = args.model_name if 'model_name' in args else None
        self.clean_file = args.clean if 'clean' in args else None

        try:
            self.preprocessed_data = self.check_path(self.path + '/' + args.clean, 'file') if self.clean_file else None
            self.preprocessed_data = self.preprocessed_data[self.preprocessed_data.rfind('/') + 1:] \
                if self.clean_file else None
            self.preprocessed_data = pd.read_csv(self.path + self.preprocessed_data) if self.preprocessed_data else None
        except OSError:
            self.preprocessed_data = self.check_path(self.path + '/clean/' + args.clean,
                                                     'file') if 'clean' in args else None
            self.preprocessed_data = self.preprocessed_data[
                                     self.preprocessed_data.rfind('/') + 1:] if self.clean_file else None
            self.preprocessed_data = pd.read_csv(self.path + '/clean/' + self.preprocessed_data) \
                if self.preprocessed_data else None
        self.value_conversion = {}
        # data after unclassified rows deletion and blank cells filled
        self.reduced_data_b4_change = None
        self.result_save_name = args.result_name if 'result_name' in args else None

    def check_path(self, path=None, message='directory'):
        """
        This function checks if a given directory is valid.
        if not, prompts a message to accept a valid directory from the user.
        :param path: the directory to check
        :type path: str
        :param message: the message to display
        :type message: str
        :return: a valid directory
        :rtype: str
        """
        path = self.path if path is None else path
        path = path.replace('\\', '/')
        if not exists(path):
            raise OSError(f"Invalid {message}.\nPlease enter a valid {message}.")
        return path

    def test_data_and_preprocessing(self, data=None, save_name=None):
        """
        This method runs all the data preprocessing steps
        :param data: the data to preprocess
        :type data: DataFrame
        :return: the preprocessed data
        :rtype: DataFrame
        """
        save_name = save_name if save_name else self.save_name if self.save_name else self.file
        data = self.data if data is None else data
        data = self.delete_unclassified_rows(data=data)
        data = self.fill_blanks(data=data)
        self.reduced_data_b4_change = data
        data = self.normalize(data=data)
        data = self.discretize(data=data)
        self.preprocessed_data = data

        preprocessing_conversions = Container(value_conversion=self.value_conversion,
                                              scale=self.scale,
                                              normalization=self.normalization,
                                              reduced_data_b4_change=self.reduced_data_b4_change,
                                              preprocessed_data=self.preprocessed_data,
                                              data=self.data)

        if not os.path.exists(self.path + 'clean/'):
            os.mkdir(self.path + 'clean/')

        addition = '' if save_name != self.file else '_clean'
        self.save_result(file=save_name, path=self.path + 'clean/', data=data, addition=addition)

        save_name = save_name[:save_name.rfind('.')]
        save_name += '_clean_preprocessing_conversions'

        with open(self.path + 'clean/' + save_name, 'wb') as file:
            pickle.dump(preprocessing_conversions, file)

        print(f"Preprocessed train data + label conversions saved at:\n\t{self.path + 'clean/'}")

        return data

    def build_model(self, data=None, file_name=None, algorithm_type=None, classifier=None, clean_file=None):
        """
        This method builds the model based on the chosen algorithm and saves it to a pickle
        :param data: the data to build the model upon
        :type data: DataFrame
        :return: the built model after training
        """
        # load model preprocessing data
        if not self.value_conversion or self.reduced_data_b4_change is None:
            self.load_preprocessing_file(clean_file[:clean_file.rfind('.csv')])
        # initiate variables
        data = self.preprocessed_data if data is None else data
        file_name = file_name if file_name else 'trained_model_' + self.algorithm \
            if self.model_name is None else self.model_name
        clean_file = clean_file if clean_file else self.clean_file
        model = self.apply_model_algorithm(data=data, algorithm_type=algorithm_type, classifier=classifier)

        data_predict = data.copy()
        data_predict.replace(self.value_conversion, inplace=True)

        # gets the training data predictions
        predictions = self.model.predict(data_predict.drop(self.classifier, axis=1))

        # in case of a clustering algorithm
        cluster_classifier_convert = {}
        if isinstance(model, KMeans):
            # builds a dictionary for converting clusters into their most common classifier
            data_predict['cluster'] = model.labels_
            for cluster in data_predict['cluster'].unique():
                cluster_classifier_convert[cluster] = \
                    data_predict[data_predict['cluster'] == cluster][self.classifier].value_counts().idxmax()
            data_predict.drop('cluster', axis=1, inplace=True)
            # converts the predictions from clusters to classifiers
            predictions = [cluster_classifier_convert[prediction] for prediction in predictions]
            model = ClassifierContainer(self.model,
                                        converter=cluster_classifier_convert,
                                        value_conversion=self.value_conversion,
                                        normalization=self.normalization,
                                        scale=self.scale,
                                        data=self.data)
        else:
            model = ClassifierContainer(self.model,
                                        value_conversion=self.value_conversion,
                                        normalization=self.normalization,
                                        scale=self.scale,
                                        data=self.data)

        # builds a dictionary to reverse the value conversion of the classifiers
        flip_classifiers = {v: k for k, v in self.value_conversion[self.classifier].items()}

        if not os.path.exists(self.path + 'models/'):
            os.mkdir(self.path + 'models/')
        with open(self.path + 'models/' + file_name, 'wb') as file:
            pickle.dump(model, file)
        self.model = model

        # adds the predictions with the original classifiers to the data before normalization and discretization
        self.reduced_data_b4_change['prediction'] = [flip_classifiers[value] for value in predictions]
        # adds a folder for the predictions (if it doesn't exist) in the given data path
        if not os.path.exists(self.path + 'predictions/'):
            os.mkdir(self.path + 'predictions/')
        # saves the train data with it's predictions with the original values
        self.save_result(data=self.reduced_data_b4_change, path=self.path + 'predictions/', file=self.clean_file,
                         addition='_prediction_' + self.algorithm)
        print(f"Predictions saved at:\n\t{self.path + 'predictions/'}")

        print("Train ", end='')
        precision_check(self.reduced_data_b4_change[self.classifier], self.reduced_data_b4_change['prediction'])

        print(f"{self.algorithm} model saved at:\n\t{self.path + 'models/' + file_name}")

        return model

    def run_model(self, data=None, model=None, model_name=None):
        data = data if data else self.test_data
        model = self.model if model is None else model
        model_name = model_name if model_name \
            else self.model_name if self.model_name \
            else self.save_name if self.save_name else self.file[:self.file.rfind('.')] if self.file else ''
        if not model:
            try:
                with open(self.path + 'models/' + model_name, 'rb') as save_model:
                    model = pickle.load(save_model)
            except OSError:
                with open(self.path + model_name, 'rb') as save_model:
                    model = pickle.load(save_model)

        if not (self.normalization is not None
                and self.scale
                and self.value_conversion) \
                and isinstance(model, ClassifierContainer):
            self.normalization = model.get_extra_by_name('normalization')
            self.scale = model.get_extra_by_name('scale')
            self.value_conversion = model.get_extra_by_name('value_conversion')
            self.data = model.get_extra_by_name('data')

        test_data_raw = copy.deepcopy(data)
        test_data = test_data_raw.copy()

        test_data = self.reverse_conversion(test_data)
        test_data.replace(self.value_conversion, inplace=True)
        # gets the predictions
        predictions = model.predict(test_data.drop(self.classifier, axis=1))
        # in case of a clustering algorithm
        if isinstance(model, ClassifierContainer) and isinstance(model.get_classifier_algorithm(), KMeans):
            cluster_classifier_convert = model.get_extra_by_name('converter')
            predictions = [cluster_classifier_convert[prediction] for prediction in predictions]
        # adds the prediction to the original test data with the original values
        flip_classifiers = {v: k for k, v in self.value_conversion[self.classifier].items()}
        test_data_raw['prediction'] = [flip_classifiers[value] for value in predictions]

        if not self.result_save_name:
            self.save_result(data=test_data_raw,
                             file='test.csv',
                             path=self.path + 'predictions/',
                             addition='_prediction_' + model.__class__.__name__)
        else:
            self.save_result(data=test_data_raw,
                             path=self.path + 'predictions/',
                             file=self.result_save_name)
        print("Test ", end='')
        precision_check(test_data_raw[self.classifier], test_data_raw['prediction'])

        return test_data_raw

    def reverse_conversion(self, data=None, test_data=None):
        test_data = test_data.copy() if test_data else self.test_data.copy()

        drop_columns = []
        # for converting into normalized values if normalization has been applied
        if self.normalization:
            categorical_columns = []
            non_categorical_columns = []
            for column in test_data:
                if self.is_categorical(column):
                    categorical_columns.append(column)
                else:
                    non_categorical_columns.append(column)

            temp_df = test_data.drop(categorical_columns, axis=1)
            temp_df = self.scale.transform(temp_df)
            test_data[non_categorical_columns] = temp_df

        for column in test_data:
            if column in list(self.value_conversion.keys()):
                values = test_data[column].unique()
                for value in values:
                    conversions = list(self.value_conversion[column].keys())
                    if isinstance(conversions[0], pd.Interval):
                        conversions = pd.IntervalIndex(conversions)
                        try:
                            test_data[column] = test_data[column].replace(value,
                                                                          conversions[conversions.get_loc(value)])
                        except KeyError:
                            if value > conversions[-1].right:
                                test_data[column] = test_data[column].replace(value, conversions[-1])
                            elif value <= conversions[0].left:
                                test_data[column] = test_data[column].replace(value, conversions[0])
                            else:
                                test_data[column] = test_data[column].replace(value, 0)
                    elif value not in conversions:
                        test_data[column] = test_data[column].replace(value, 0)
            else:

                drop_columns.append(column)

        test_data.drop(drop_columns, axis=1, inplace=True)

        return test_data

    def load_preprocessing_file(self, model_name=''):
        if 'clean' in model_name:
            model_name = model_name[:model_name.rfind('_clean')]
        model_name += '_clean_preprocessing_conversions'
        preprocessing_conversions = None
        try:
            # self.check_path(self.path + 'clean/' + model_name, 'model file')
            with open(self.path + 'clean/' + model_name, 'rb') as pp_container:
                preprocessing_conversions = pickle.load(pp_container)
        except OSError:
            # self.check_path(self.path + '/' + model_name, 'model file')
            with open(self.path + model_name, 'rb') as pp_container:
                preprocessing_conversions = pickle.load(pp_container)

        if preprocessing_conversions:
            self.normalization = preprocessing_conversions.get_extra_by_name('normalization')
            self.scale = preprocessing_conversions.get_extra_by_name('scale')
            self.value_conversion = preprocessing_conversions.get_extra_by_name('value_conversion')
            self.reduced_data_b4_change = preprocessing_conversions.get_extra_by_name('reduced_data_b4_change')
            self.data = preprocessing_conversions.get_extra_by_name('data')

        return preprocessing_conversions

    def delete_unclassified_rows(self, data=None, classifier=None):
        """
        This method deletes rows which are unclassified (NaN cells in classifier column)
        :param data: the data frame to remove the rows from (if not given, will be the instance's defined data frame)
        :type data: DataFrame
        :param classifier: the classifier column (if not given, will be the instance's defined classifier column)
        :type classifier: str or DataFrame
        :return: the data frame after unclassified rows deletion
        :rtype: DataFrame
        """
        data = self.data if data is None else data
        classifier = self.classifier if classifier is None else classifier
        return data.dropna(subset=[classifier])

    def fill_blanks(self, fill_type=None, data=None):
        """
        This method fills the blank cells of the data frame based on all of the data or the classifier column value
        mean for continuous values and most common value for categorical values
        :param fill_type: the method to fill the empty cells - based on classifier or based on all of the data
        :type fill_type: str
        :param data: DataFrame
        :return: the data frame after filling the empty cells
        :rtype: DataFrame
        """

        fill_type = self.fill_type if fill_type is None else fill_type
        data = self.data if data is None else data
        return_data = data.copy()

        # fill blank cells according to all the data set
        if fill_type in {'all', 'All', 'ALL', 'a', 'A'}:
            for column in return_data:
                if not self.is_categorical(column):
                    parameter = return_data[column].mean()
                else:
                    parameter = return_data[column].value_counts().idxmax()
                return_data[column].fillna(value=parameter, inplace=True)

            print('discrete according to all of the data')
            pass
        # fill blank cells according to classifier
        else:
            for column in return_data.columns:
                for classifier in return_data[self.classifier].unique():
                    if not self.is_categorical(column):
                        parameter = return_data.loc[return_data[self.classifier] == classifier, column].mean()
                    else:
                        parameter = return_data.loc[
                            return_data[self.classifier] == classifier, column].value_counts().idxmax()
                    return_data.loc[return_data[self.classifier] == classifier, column] = \
                        return_data.loc[return_data[self.classifier] == classifier, column].fillna(value=parameter)
            print('discrete according to the classifier')
        return return_data

    def normalize(self, data=None, apply_normalization=None):
        """
        The function performs normalization
        :param data: the data frame to remove the rows from (if not given, will be the instance's defined data frame)
        :type data: DataFrame
        :param apply_normalization: Will be apply normalization
        :type apply_normalization: bool or str
        :return: the data after normalization  or the data without normalization
        :rtype : str, DataFrame
        """

        data = self.data if data is None else data
        apply_normalization = self.normalization if apply_normalization is None else apply_normalization
        # apply normalization if user according to flag (user request)
        if apply_normalization in {True, 'true', 'True', 'TRUE', 't', 'T', 'y', 'Y', 'yes', 'Yes', 'YES'}:
            print('apply normalization')
            return_data = data.copy()
            categorical_columns = []
            non_categorical_columns = []
            for column in return_data:
                if self.is_categorical(column):
                    categorical_columns.append(column)
                else:
                    non_categorical_columns.append(column)
            temp_df = return_data.drop(categorical_columns, axis=1)
            temp_df = self.scale.fit_transform(temp_df)
            return_data[non_categorical_columns] = temp_df
            return return_data
        else:
            print('do not apply normalization')

        return data

    def discretize(self, data=None, discretization_type='_', bins=None, classifier=None):
        """
        The function performs discretization
        :param data: the data frame to remove the rows from (if not given, will be the instance's defined data frame)
        :type data: DataFrame
        :param discretization_type: the type of the discretization
        :type discretization_type: str
        :param bins: the number (if integer is given) or sequence of bins to divide the data to
        :type bins: int or Sequence
        :return: the data after discretization
        :rtype : str, DataFrame
        """

        discretization_type = self.discretization if discretization_type == '_' else discretization_type
        data = self.data if data is None else data
        bins = bins if bins else self.bins if self.bins else 6
        return_data = data.copy()

        if isinstance(bins, Sequence):
            bins = len(bins)

        # equal depth (frequency) discretization
        if discretization_type in {'equal_depth', 'Equal_Depth', 'EqualDepth', 'ed', 'Ed', 'ED', 'e.d', 'E.d', 'E.D',
                                   'equal_frequency', 'Equal_Frequency', 'EqualFrequency', 'ef', 'Ef', 'EF',
                                   'e.f', 'E.f', 'E.F', 'e_d', 'E_d', 'E_D', 'e_f', 'E_f', 'E_F'}:
            for column in return_data:
                if not self.is_categorical(column):
                    if self.implementation:
                        return_data[column] = EqualDepthDiscretization(convert_to_list(return_data[column]), bins=bins) \
                            .get_with_labels()
                    else:
                        return_data[column] = pd.qcut(np.array(return_data[column]), bins, duplicates='drop')
            print('equal depth discretization')

        # equal width discretization
        elif discretization_type in {'equal_width', 'Equal_Width', 'EqualWidth', 'ew', 'Ew', 'EW', 'e.w', 'E.w', 'E.W',
                                     'e_w', 'E_w', 'E_W'}:

            for column in return_data:
                if not self.is_categorical(column):
                    if self.implementation:
                        return_data[column] = EqualWidthDiscretization(convert_to_list(return_data[column]), bins=bins) \
                            .get_with_labels()
                    else:
                        return_data[column] = pd.cut(np.array(return_data[column]), bins)
            print('equal width discretization')

        # entropy based discretization
        elif discretization_type in {'entropy', 'Entropy', 'en', 'En', 'EN', 'e.n', 'E.n', 'E.N', 'e_n', 'E_n', 'E_N'}:
            _INFO_GAIN_RATIO_THRESHOLD = 0.001
            classifier = self.classifier if classifier is None else classifier

            for column in return_data.columns:
                if not self.is_categorical(column):
                    return_data[column] = EntropyDiscretization(return_data[column],
                                                                return_data[self.classifier],
                                                                self.bins,
                                                                minimal_gain=_INFO_GAIN_RATIO_THRESHOLD). \
                        apply_discretization()

            print('entropy based discretization')
        else:
            print('no discretization')

        # data conversion dictionary for applying the same preprocessing on future/test data (starts at o1 to leave 0
        # for values which were not in the train dataset
        for column in return_data.columns:
            self.value_conversion[column] = {value: index for index, value in
                                             enumerate(return_data[column].sort_values().unique(), start=1)}
        return_data.replace(self.value_conversion, inplace=True)
        return return_data

    def apply_model_algorithm(self, data=None, algorithm_type=None, classifier=None):
        data = self.preprocessed_data if data is None else data
        algorithm_type = self.algorithm if algorithm_type is None else algorithm_type
        classifier = self.classifier if classifier is None else classifier
        return_data = data.copy()

        algorithm = None
        # for the case that own implementation is chosen
        if self.implementation:
            for alg in classification_algorithms.__algorithms__:
                if alg().algorithm_by_name(algorithm_type):
                    if isinstance(alg(), DecisionTreeID3):
                        weight = 0.0001 if len(data.index) > 100 else 0
                        min_samples = int(len(data.index) * weight) if len(data.index) > 1000 else 1
                        algorithm = DecisionTreeID3(min_gain=0.03,
                                                    min_samples_leaf=350,
                                                    max_depth=25)
                    else:
                        algorithm = alg()
                    break
        # for the case that own implementation is not chosen or algorithm wasn't found in own implementations
        # e.g. use external implementation (sklearn)
        if not algorithm:
            if algorithm_type in {'decision_tree', 'Decision_Tree', 'DecisionTree', 'Decision_tree', 'decision',
                                  'Decision', 'd', 'D', 'dt', 'DT', 'd_t', 'D_T', 'd.t', 'D.T', 'Dt', 'D_t', 'D.t'}:

                # entropy based splitting to avoid over-fitting:
                # 1. with min weight (fraction) of each node as 0.5% if samples are larger than 20
                # 2. minimum samples for each node is 30 if samples are above 1000
                weight = 0.0001 if len(data.index) > 100 else 0
                min_samples = int(len(data.index) * weight) if len(data.index) > 1000 else 1
                if self.implementation:
                    algorithm = DecisionTreeID3(min_gain=0.02,
                                                min_samples_leaf=min_samples,
                                                max_depth=25)
                else:
                    algorithm = DecisionTreeClassifier(criterion='entropy',
                                                       min_weight_fraction_leaf=weight,
                                                       min_samples_leaf=min_samples)

            elif algorithm_type in {'naive_bayes', 'Naive_Bayes', 'NaiveBayes', 'Naive_bayes', 'Naive', 'naive', 'n',
                                    'N', 'nb', 'NB', 'n_b', 'N_B', 'n.b', 'N.B', 'Nb', 'N_b', 'N.b'}:
                # checks if the data was dicretisized
                if self.discretization:
                    algorithm = CategoricalNB()
                # applies the Gaussian naive base since the data may not be discrete
                else:
                    algorithm = GaussianNB()

            elif algorithm_type in {'KNeighbors', 'kneighbors', 'K_Neighbors', 'K_neighbors', 'k_neighbors',
                                    'k_Neighbors',
                                    'KNN', 'kNN', 'KNn', 'Knn', 'knn', 'K_N_N', 'K_N_n', 'K_n_n', 'k_n_n', 'K.N.N',
                                    'K.N.n',
                                    'K.n.n', 'k.n.n', 'KN', 'kn', 'K_N', 'K_n', 'k_N', 'k_n', 'K.N', 'K.n', 'k.N',
                                    'k.n'}:
                algorithm = KNeighborsClassifier()
            elif any(
                    algorithm_type.casefold() == name.casefold() for name in {'kmeans', 'km', 'k.m', 'k_m', 'k_means'}):
                algorithm = KMeans(n_clusters=15)
            # for the case that algorithm wasn't found in built in implementations - looks in own implementations
            elif algorithm_type:
                for alg in classification_algorithms.__algorithms__:
                    if alg().algorithm_by_name(name=algorithm_type):
                        algorithm = alg
                        break
        if not algorithm:
            raise TypeError('algorithm type not found.')

        self.model = algorithm.fit(return_data.drop([classifier], axis=1), return_data[classifier])
        return self.model

    def save_result(self, data=None, path=None, file=None, addition='', suffix=None):
        """
        This method saves given data (or instance's data if not given) to a csv file (adding '_clean' to the file name)
        :param data: the data to save
        :type data: DataFrame
        :param file: the name of the original file
        :type file: str
        :param path: the path (directory) where to save the file
        :type path: str
        :return: nothing
        :rtype: None
        """
        data = self.data if data is None else data
        path = self.path if path is None else path
        file_name = self.file if file is None else file

        suffix_ = '.' if suffix and '.' not in suffix else ''
        suffix_ += suffix if suffix else ''
        if '.' in file_name and not suffix:
            file_name, suffix_ = file_name.split('.')
            suffix_ = '.' + suffix_

        data.to_csv(path + file_name + addition + suffix_, index=False)

    def is_categorical(self, column_, data=None, path=None):
        """
        This function checks if a given column has data which is categorical (or continuous)
        :param column_: the column to check if the data in it is categorical
        :type column_: str or DataFrame
        :param data: the data frame to remove the rows from (if not given, will be the instance's defined data frame)
        :type data: DataFrame
        :param path: The path to the data files
        :type path: str
        :return: true if the column is categorical (false if continuous)
        :rtype: bool
        """
        # both following constants are used for determining
        # if the values in a column is categorical (otherwise continuous)
        # 5% threshold for the ratio between unique values and the total values of a column.
        _RATIO_THRESHOLD = 0.05
        # maximum number of unique values within a column
        _MAX_CATEGORICAL_SIZE = 25

        data = data if data else self.data
        path = path if path else self.path

        if data is None:
            return
        try:
            with open(path + "Structure.txt", "r") as structure_file:
                for line in structure_file:
                    if column_ in line and 'NUMERIC' in line:
                        return False
                return True
        except OSError:
            pass
        size = data[column_].nunique()

        return data.dtypes[column_].name == 'category' \
               or data.dtypes[column_].name == 'object' \
               or 1.0 * size / data[column_].count() < _RATIO_THRESHOLD \
               and size <= _MAX_CATEGORICAL_SIZE

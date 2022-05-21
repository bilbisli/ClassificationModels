"""
Author:
            Name                Email
        Israel Avihail      bilbisli@gmail.com
"""

import argparse
from argparse import RawTextHelpFormatter
import classification_algorithms
from data_analysis import DataAnalysis


def main(parser=None):
    built_in_algorithms = ['naive_bayes', 'decision_tree', 'k_neighbors', 'k_means']
    non_built_in = [str(max(alg().get_strings(), key=len)).lower() for alg in classification_algorithms.__algorithms__
                    if not any(str(max(alg().get_strings(), key=len)).casefold() == name.casefold()
                               for name in built_in_algorithms)]

    parser = argparse.ArgumentParser(
        # prog="DataAnalysis",
        description="Description: Analyse data with data mining tools.",
        epilog='Made by Israel Avihail.\n'
               + 'For bugs & issues: bilbisli@gmail.com',
        formatter_class=RawTextHelpFormatter,
        add_help=True,
    ) if not parser else parser

    subparsers = parser.add_subparsers(
        help='run mode help',
        title='run modes',
        description='run modes define which mode the system should run in the current execution.\n'
                    'example: classification_models.py all train.csv test.csv C:/Users/user/Desktop/data\n'
                    '\t\tor\n'
                    '\t   classification_models.py preprocessing train.csv ./resources\n',
        metavar='WORKING_MODE',
    )

    parser.add_argument(
        'path',
        type=str,
        help='Path directory of dataset files.\nexample: C:/Users/user/Desktop/data\n\t\tor\n\t ./resources',
        metavar="DATA_PATH",
    )

    #preprocessing case:
    preprocessing = subparsers.add_parser('preprocessing',
                                          help='in this mode only preprocessing is applied',
                                          aliases=['p', 'pp'],
                                          formatter_class=RawTextHelpFormatter,
                                          )
    preprocessing.set_defaults(work_mode='preprocessing')
    preprocessing.add_argument(
        'train',
        type=str,
        help='Training dataset file name. example: train.csv',
        metavar="TRAINING_FILE_NAME",
    )
    preprocessing.add_argument(
        '--fill',
        dest='fill',
        type=str,
        help='Fill blank cells parameter. example: --fill all',
        metavar="FILL_BLANKS_TYPE",
        default='classifier',
    )
    preprocessing.add_argument(
        '--normalization',
        dest='normalization',
        action='store_true',
        help='Apply normalization. example: --normalization',
    )
    preprocessing.add_argument(
        '--no-normalization',
        dest='normalization',
        action='store_false',
        help='Do not apply normalization. example: --no-normalization',
    )
    preprocessing.set_defaults(normalization=False)
    preprocessing.add_argument(
        '--discretization',
        dest='discretization',
        type=str,
        help='Discretization type. example: --discretization equal_width',
        metavar="DISCRETIZATION_TYPE",
    )
    preprocessing.add_argument(
        '--bins',
        dest='bins',
        nargs='+',
        # type=str,
        help='Number of bins (intervals) the continues data will be divided to. example: --bins 5',
        metavar="BIN_NUMBER",
        default=[],
    )
    preprocessing.add_argument(
        '--implementation',
        dest='implementation',
        type=str,
        help='Apply built in/own implementations of classifying/discretization algorithms(if exists).\n'
             'example: --implementation own',
        choices=['own', 'built_in'],
        metavar="IMPLEMENTATION_TYPE",
        default="built_in",
    )
    preprocessing.add_argument(
        '--save_name',
        dest='save_name',
        type=str,
        help='The name of the file to be saved after processing. example: name.csv',
        metavar="FILE_NAME",
    )
    #build model case:
    build_model = subparsers.add_parser('build_model',
                                        help='in this mode only model build is is done',
                                        aliases=['bm'],
                                        formatter_class=RawTextHelpFormatter,
                                        # parents=[parser],
                                        )
    build_model.set_defaults(work_mode='build_model')
    build_model.add_argument(
                'clean',
                type=str,
                help='Training dataset file name (already undergone preprocessing). example: train_clean.csv',
                metavar="POST_PREPROCESSED_FILE_NAME",
            )
    build_model.add_argument(
        '--algorithm',
        dest='algorithm',
        type=str,
        help='Model algorithm type. example: --algorithm algorithm_type.\noptions: '
             + ', '.join(built_in_algorithms + non_built_in),
        metavar="ALGORITHM_TYPE",
        default="dt",
    )
    build_model.add_argument(
        '--implementation',
        dest='implementation',
        type=str,
        help='Model algorithm type. example: --implementaion built_in',
        choices=['own', 'built_in'],
        metavar="IMPLEMENTATION_TYPE",
        default="built_in",
    )
    build_model.add_argument(
        '--model_name',
        dest='model_name',
        type=str,
        help='The name of the model to be saved (as pickle). example: --model_name decision_tree_model_1',
        metavar="MODEL_NAME",
    )

    # run model case:
    run_model = subparsers.add_parser('run_model',
                                      help='in this mode the only operation done is running a model on test data',
                                      aliases=['rm', 'r'],
                                      # parents=[parser],
                                      )
    run_model.set_defaults(work_mode='run_model')
    preprocessing.set_defaults(normalization=None)
    run_model.add_argument(
        'test',
        type=str,
        help='Test dataset file name. example: test.csv',
        metavar="TEST_FILE_NAME",
    )
    run_model.add_argument(
        '--model_name',
        dest='model_name',
        type=str,
        help='Model file name that is already saved (as pickle). example: --model_name decision_tree_model_1',
        metavar="TEST_FILE_NAME",
        required=True,
    )
    run_model.add_argument(
        '--result_name',
        dest='result_name',
        type=str,
        help='Prediction result file name to save. example: --result_name test_predicition_DecisionTree_1.csv',
        metavar="PREDICTION_RESULT_FILE_NAME",
    )
    all_process = subparsers.add_parser('all',
                                        help='in this mode the whole program will be executed',
                                        aliases=['ALL', 'a', 'A'],
                                        formatter_class=RawTextHelpFormatter
                                        )
    all_process.add_argument(
        'train',
        type=str,
        help='Training dataset file name. example: train.csv',
        metavar="TRAINING_FILE_NAME",
    )
    all_process.add_argument(
        'test',
        type=str,
        help='Test dataset file name. example: test.csv',
        metavar="TEST_FILE_NAME",
    )
    all_process.add_argument(
        '--fill',
        dest='fill',
        type=str,
        help='Fill blank cells parameter. example: --fill all',
        metavar="FILL_BLANKS_TYPE",
        default='classifier',
    )
    all_process.add_argument(
        '--normalization',
        dest='normalization',
        action='store_true',
        help='Apply normalization. example: --normalization',
    )
    all_process.add_argument(
        '--no-normalization',
        dest='normalization',
        action='store_false',
        help='Do not apply normalization. example: --no-normalization',
    )
    all_process.set_defaults(normalization=False)
    all_process.add_argument(
        '--discretization',
        dest='discretization',
        type=str,
        help='Discretization type. example: --discretization equal_width',
        metavar="DISCRETIZATION_TYPE",
    )
    all_process.add_argument(
        '--bins',
        dest='bins',
        nargs='+',
        # type=str,
        help='Number of bins (intervals) the continues data will be divided to. example: --bins=5',
        metavar="BIN_NUMBER",
        default=[6],
    )
    all_process.add_argument(
        '--algorithm',
        dest='algorithm',
        type=str,
        help='Model algorithm type. example: --algorithm algorithm_type\noptions: '
             + ', '.join(built_in_algorithms + non_built_in),
        metavar="ALGORITHM_TYPE",
        default="dt",
    )
    all_process.add_argument(
        '--implementation',
        dest='implementation',
        type=str,
        help='Apply built in/own implementations of classifying/discretization algorithms(if exists).\n'
             'example: --implementation own',
        choices=['own', 'built_in'],
        metavar="IMPLEMENTATION_TYPE",
        default="built_in",
    )
    all_process.add_argument(
        '--result_name',
        dest='result_name',
        type=str,
        help='Prediction result file name to save. example: --result_name test_predicition_DecisionTree_1.csv',
        metavar="PREDICTION_RESULT_FILE_NAME",
    )

    args = parser.parse_args()
    data_analysis = DataAnalysis(args)

    if 'work_mode' in args and args.work_mode in {'pre-processing', 'pre_processing', 'preprocessing', 'pp', 'PP', 'P', 'p'}:
        data_analysis.test_data_and_preprocessing()
    elif 'work_mode' in args and args.work_mode in {'build-model', 'build_model', 'BM', 'bm', 'b', 'B'}:
        data_analysis.build_model()
    elif 'work_mode' in args and args.work_mode in {'run_model', 'test', 't', 'T'}:
        data_analysis.run_model()
    else:
        data_analysis.test_data_and_preprocessing()
        data_analysis.build_model()
        data_analysis.run_model()


if __name__ == '__main__':
    main()



# ClassificationModels
Analyse data with data mining tools using this program.
### Author:
Israel Avihail	

## Required libraries:
### import of file data_analysis
 - argparse
the object will hold all the information necessary to parse the command line into Python data types.
 - on
os.path.exists Used to check if the path exists, if isn't exists we use os.mkdir to create.
 - pickle
it serializes objects so they can be saved to a file, and loaded in a program again later on.
 - Sequence
used for check if the object is type of Sequence.
 - operator.itemgetter
return to us item all the time we use in genrator.
 - numpy
numpy is a Python library used for working with arrays
- pandas
uesd to read csv files and do operations on it, Intervals used for us to check if object is the same type, pd.IntervalIndex use to convert to interval
 - stats
We import scipy.stats to use the entropy function which represents the effective size index of probability space.
 - sklearn
Kmeans, GaussianNB, CategoricalNB, KNeighborsClassifier, StandardScaler:  used for calculation and generation of Confusion Matrix PDFs
### import of file Classifier_Algorithm
 - abc
This module provides the infrastructure for defining abstract base classes (ABCs) in Python, we used to do interface

### import of file dictionary_tree

 - copy
import copy to use for deep copy

 - convert_to_list
This function converts given data to a list

### import of file entropy

 - Fraction
used to convert two numbers to number rational and we used to check if object is the same type of fraction
 - log2
used to calculate entropy

### import of file entropy_discretization

- combinations
It return r-length tuples in sorted order with no repeated elements we use in entropy with genretor his return one tuples from the list

## How to add custom classification algorithm:
step 1: add the algorithem to package/folder of "classification_algorithms" within the project
step 2: in the package/folder "classification_algorithms" within the "__init__.py" file:
	 2.1: import the new algorithm (class). example: "from classification_algorithms.algorithm_file import AlgorithmName"
	 2.2: add the algorithm (class) to "__algorithm__" list. example: "[AlgorithmName, <existing algorithms...>]"
*It is recommended that the classification algorithm will implement the classification algorithm interface "ClassifierAlgorithm" which is located in "project_util" package/folder.
example: 
	from project_util.classifier_algorithm import ClassifierAlgorithm
	class AlgorithmName(ClassifierAlgorithm):
  
## Preparing and running within a Virtual Enviroment:

##### -- option 1 --
	<ul>
		<li>step 1: create a virtual enviroment (python must be installed before hand)</li>
		<ul>
		<li>1.1. open a shell within the project folder ("ClassifyingModels")</li>
		<li>1.1.1 on Windows run the command:  py -m venv env</li>
		<li>1.1.2 on Unix/MacOs run the command: python3 -m venv env</li>
		</ul>
	<li>step 2: activate the enviroment</li>
		<ul>
		<li>2.1 on Windows run the command: .\env\Scripts\activate</li>
		<li>2.2 on Unix/MacOs run the command: source env/bin/activate</li>
		</ul>
	<li>step 3: install requirements</li>
		<ul>
		<li>3.1 on Windows run the command: py -m pip install -r requirements.txt</li>
		<li>3.2 on Unix/MacOs run the command: python3 -m pip install -r requirements.txt</li>
		</ul>
	</ul>
	
##### -- option 2 --
	1. on Windows run the bat file within the project folder: "run_windows.bat"
	2. on Unix/MacOs (or Windows with supporting shell such as git) run the sh file withing the project folder: "run_unix_mac.sh"

When the enviroment is all set, the program can now be run.

Important!!
After either options, when work with the program is finished, the enviroment needs to be deactivated:
	In the active enviroment's open shell run the command: deactivate
	
## How to Run :
#### Command help section - To see this text (the help section) via the program  - run the command (in the open shell): classification_models.py -h

usage: classification_models.py [-h] WORKING_MODE ... DATA_PATH

Description: Analyse data with data mining tools.

positional arguments:
  DATA_PATH             Path directory of dataset files.
                        example: C:/Users/user/Desktop/data
                                        or
                                 ./resources

options:
  -h, --help            show this help message and exit

run modes:
  run modes define which mode the system should run in the current execution.
  example: classification_models.py all train.csv test.csv C:/Users/user/Desktop/data
                or
           classification_models.py preprocessing train.csv ./resources

  WORKING_MODE          run mode help
    preprocessing (p, pp)
                        in this mode only preprocessing is applied
    build_model (bm)    in this mode only model build is is done
    run_model (rm, r)   in this mode the only operation done is running a model on test data
    all (ALL, a, A)     in this mode the whole program will be executed

Made by Israel Avihail.
For bugs & issues: bilbisli@gmail.com

#### Build Model help section - To see this text (the help section) via the program - run the command (in the open shell): classification_models.py bm -h
usage: classification_models.py build_model [-h] [--algorithm ALGORITHM_TYPE] [--implementation IMPLEMENTATION_TYPE]
                                            [--model_name MODEL_NAME]
                                            POST_PREPROCESSED_FILE_NAME

positional arguments:
  POST_PREPROCESSED_FILE_NAME
                        Training dataset file name (already undergone preprocessing). example: train_clean.csv

options:
  -h, --help            show this help message and exit
  --algorithm ALGORITHM_TYPE
                        Model algorithm type. example: --algorithm algorithm_type.
                        options: naive_bayes, decision_tree, k_neighbors, k_means
  --implementation IMPLEMENTATION_TYPE
                        Model algorithm type. example: --implementaion built_in
  --model_name MODEL_NAME
                        The name of the model to be saved (as pickle). example: --model_name decision_tree_model_1

#### Run Model help section - To see this text (the help section) via the program - run the command (in the open shell): classification_models.py rm -h

positional arguments:
  TEST_FILE_NAME        Test dataset file name. example: test.csv

options:
  -h, --help            show this help message and exit
  --model_name TEST_FILE_NAME
                        Model file name that is already saved (as pickle). example: --model_name decision_tree_model_1
  --result_name PREDICTION_RESULT_FILE_NAME
                        Prediction result file name to save. example: --result_name test_predicition_DecisionTree_1.csv


#### Run Mode help section (example for 'all') - To see this text (the help section) via the program - run the command (in the open shell): classification_models.py a -h

usage: classification_models.py all [-h] [--fill FILL_BLANKS_TYPE] [--normalization] [--no-normalization]
                                    [--discretization DISCRETIZATION_TYPE] [--bins BIN_NUMBER [BIN_NUMBER ...]]
                                    [--algorithm ALGORITHM_TYPE] [--implementation IMPLEMENTATION_TYPE]
                                    [--result_name PREDICTION_RESULT_FILE_NAME]
                                    TRAINING_FILE_NAME TEST_FILE_NAME

positional arguments:
  TRAINING_FILE_NAME    Training dataset file name. example: train.csv
  TEST_FILE_NAME        Test dataset file name. example: test.csv

options:
  -h, --help            show this help message and exit
  --fill FILL_BLANKS_TYPE
                        Fill blank cells parameter. example: --fill all
  --normalization       Apply normalization. example: --normalization
  --no-normalization    Do not apply normalization. example: --no-normalization
  --discretization DISCRETIZATION_TYPE
                        Discretization type. example: --discretization equal_width
  --bins BIN_NUMBER [BIN_NUMBER ...]
                        Number of bins (intervals) the continues data will be divided to. example: --bins=5
  --algorithm ALGORITHM_TYPE
                        Model algorithm type. example: --algorithm algorithm_type
                        options: naive_bayes, decision_tree, k_neighbors, k_means
  --implementation IMPLEMENTATION_TYPE
                        Apply built in/own implementations of classifying/discretization algorithms(if exists).
                        example: --implementation own
  --result_name PREDICTION_RESULT_FILE_NAME
                        Prediction result file name to save. example: --result_name test_predicition_DecisionTree_1.csv

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime
import os
import logging

dir_name = os.path.basename(__file__).removesuffix(".py")
filename = "log " + datetime.now().strftime("%m-%d-%Y %H-%M-%S") + ".txt"
log_file = os.path.join(dir_name, filename)
os.makedirs(dir_name, exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.DEBUG,
                    format="%(asctime)s %(message)s")

def parseNumber(s):
    try:
        return float(s)
    except ValueError:
        return s

data_train = np.loadtxt('./KDDTrain+.txt', dtype =object, delimiter=',', encoding='latin1', converters=parseNumber)
data_test = np.loadtxt('./KDDTest+.txt', dtype =object, delimiter=',', encoding='latin1', converters=parseNumber)
print('len(data_train)', len(data_train))
print('len(data_test)', len(data_test))
logging.info('len(data_train): %s', len(data_train))
logging.info('len(data_test): %s', len(data_test))

X_train_raw = data_train[:, 0:41]
y_train_raw = data_train[:, [41]]
print('X_train_raw[0:3]===========', X_train_raw[0:3])
print('y_train_raw[0:3]===========', y_train_raw[0:3])
print('=================')

logging.info('X_train_raw[0:3]=========== \n %s', X_train_raw[0:3])
logging.info('y_train_raw[0:3]=========== \n %s', y_train_raw[0:3])

X_test_raw = data_test[:, 0:41]
y_test_raw = data_test[:, [41]]
print('X_test_raw[0:3]===========', X_test_raw[0:3])
print('y_test_raw[0:3]===========', y_test_raw[0:3])
print('=================')

logging.info('X_test_raw[0:3]=========== \n %s', X_test_raw[0:3])
logging.info('y_test_raw[0:3]=========== \n %s', y_test_raw[0:3])

x_columns = np.array(list(range(41)))
print('x_columns', x_columns)
categorical_x_columns = np.array([1, 2, 3])
numberic_x_columns = np.delete(x_columns, categorical_x_columns)
print('numberic_x_columns', numberic_x_columns)
x_ct = ColumnTransformer(transformers = [("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_x_columns),
                                         ('normalize', Normalizer(norm='l2'), numberic_x_columns)], remainder = 'passthrough')

x_ct.fit(np.concatenate((X_train_raw, X_test_raw), axis = 0))
X_train = x_ct.transform(X_train_raw)
#X_train = X_train.astype('float')
print('X_train[0:3]', X_train[0:3])
print('len(X_train[0])', len(X_train[0]))

logging.info('X_train[0:3]=========== \n %s', X_train[0:3])
logging.info('len(X_train[0]) \n %s', len(X_train[0]))

X_test = x_ct.transform(X_test_raw)
#X_train = X_train.astype('float')
print('X_test[0:3]', X_test[0:3])
print('len(X_test[0])', len(X_test[0]))

logging.info('X_test[0:3]=========== \n %s', X_test[0:3])
logging.info('len(X_test[0]) \n %s', len(X_test[0]))

categorical_y_columns = [0]
print('categorical_y_columns', categorical_y_columns)
y_ct = ColumnTransformer(transformers = [("label", OrdinalEncoder(), categorical_y_columns)], remainder = 'passthrough')

y_ct.fit(np.concatenate((y_train_raw, y_test_raw), axis = 0))
y_train = y_ct.transform(y_train_raw)
y_train = y_train.astype('float')
print('y_train[0:2]===', y_train[0:2])
(name, enc, _columns) = y_ct.transformers_[0]
print('enc.categories_', enc.categories_)
print('len(enc.categories_[0])', len(enc.categories_[0]))

logging.info('enc.categories_ %s', enc.categories_)
logging.info('len(enc.categories_[0]) %s', len(enc.categories_[0]))

y_test = y_ct.transform(y_test_raw)
y_test = y_test.astype('float')
print('y_test[0:2]===', y_test[0:2])
logging.info('y_test[0:2]===\n %s', y_test[0:2])

# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_transformed, test_size=0.2, random_state=42)

# Train a logistic regression classifier on the training set
# clf = LogisticRegression(penalty=None, C=1e-6, solver='saga', multi_class='ovr', max_iter = 100)
# clf = DecisionTreeClassifier()
params = [{
    "solver": "sgd",
    "activation": 'relu',
    "alpha": 1e-4,
    "max_iter": 200,
    "learning_rate": "constant",
    "learning_rate_init": 0.2,
    "momentum": 0.9,
    "nesterovs_momentum": False,
    "verbose": True,
    "random_state": 1,
    },
    {
    "solver": "sgd",
    "activation": 'relu',
    "alpha": 1e-4,
    "max_iter": 200,
    "learning_rate": "constant",
    "learning_rate_init": 0.1,
    "momentum": 0.9,
    "nesterovs_momentum": False,
    "verbose": True,
    "random_state": 1,
    },
    {
    "solver": "sgd",
    "activation": 'relu',
    "alpha": 1e-4,
    "max_iter": 200,
    "learning_rate": "constant",
    "learning_rate_init": 0.05,
    "momentum": 0.9,
    "nesterovs_momentum": False,
    "verbose": True,
    "random_state": 1,
    },
    {
    "solver": "sgd",
    "activation": 'relu',
    "alpha": 1e-4,
    "max_iter": 200,
    "learning_rate": "constant",
    "learning_rate_init": 0.01,
    "momentum": 0.9,
    "nesterovs_momentum": False,
    "verbose": True,
    "random_state": 1,
    },
    {
    "solver": "sgd",
    "activation": 'relu',
    "alpha": 1e-4,
    "max_iter": 500,
    "learning_rate": "constant",
    "learning_rate_init": 0.001,
    "momentum": 0.9,
    "nesterovs_momentum": False,
    "verbose": True,
    "random_state": 1,
    },
    {
    "solver": "adam",
    "activation": 'relu',
    "alpha": 1e-4,
    "max_iter": 200,
    "learning_rate": "constant",
    "learning_rate_init": 0.2,
    "momentum": 0.9,
    "nesterovs_momentum": False,
    "verbose": True,
    "random_state": 1,
    },
    {
    "solver": "adam",
    "activation": 'relu',
    "alpha": 1e-4,
    "max_iter": 200,
    "learning_rate": "constant",
    "learning_rate_init": 0.1,
    "momentum": 0.9,
    "nesterovs_momentum": False,
    "verbose": True,
    "random_state": 1,
    },
    {
    "solver": "adam",
    "activation": 'relu',
    "alpha": 1e-4,
    "max_iter": 200,
    "learning_rate": "constant",
    "learning_rate_init": 0.05,
    "momentum": 0.9,
    "nesterovs_momentum": False,
    "verbose": True,
    "random_state": 1,
    },
    {
    "solver": "adam",
    "activation": 'relu',
    "alpha": 1e-4,
    "max_iter": 200,
    "learning_rate": "constant",
    "learning_rate_init": 0.01,
    "momentum": 0.9,
    "nesterovs_momentum": False,
    "verbose": True,
    "random_state": 1,
    },
    {
    "solver": "adam",
    "activation": 'relu',
    "alpha": 1e-4,
    "max_iter": 500,
    "learning_rate": "constant",
    "learning_rate_init": 0.001,
    "momentum": 0.9,
    "nesterovs_momentum": False,
    "verbose": True,
    "random_state": 1,
    },
]
# logging.info('MLPClassifier solver = %s, activation = %s, alpha = %s, max_iter = %s, learning_rate = %s, learning_rate_init = %s',
# "sgd", 'relu', 1e-5, 200, 'constant', 0.2)
# clf = MLPClassifier(
#     hidden_layer_sizes=(41, 40, 40),
#     solver="sgd",
#     activation='relu',
#     alpha=1e-3,
#     max_iter=200,
#     learning_rate='constant',
#     learning_rate_init=0.2,
#     random_state=1,
#     verbose=True,
# )
for param in params:

    logging.info('MLPClassifier hidden_layer_sizes = %s, params= %s', (41, 20, 40), param)
    clf = MLPClassifier(
        hidden_layer_sizes=(41, 20, 40),
        **param
    )

    clf.fit(X_train, y_train.ravel())

    # Use the trained classifier to predict the classes of the test set
    y_pred = clf.predict(X_test)

    # Evaluate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    logging.info("Accuracy: %s", accuracy)
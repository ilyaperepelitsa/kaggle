
### PREPARE DATASET ###
from common_learning.source_classes import *
from common_learning.general_classes import VariableSelector
from common_learning.general_classes import LabelBinarizerPipelineFriendly
from common_learning.source_classes.preprocessing.imputer import Imputer

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import pandas as pd
import os

from collections import defaultdict

from sklearn.model_selection import train_test_split

### PIPELINE ###

from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

from scipy.stats import randint
from scipy.stats import uniform
from scipy.stats import alpha
from scipy.stats import expon
from scipy.stats import beta
from scipy.stats import gamma
from scipy.stats import betaprime
from scipy.stats import lomax

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, MultiTaskElasticNet, Lars, LassoLars, BayesianRidge, ARDRegression

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVC

from sklearn.grid_search import RandomizedSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.metrics import make_scorer

from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt
################

ames_data = pd.read_csv(
                os.path.join(
                        os.path.dirname(
                                os.path.abspath("__file__")
                        ),'amnes/data/train.csv'
                    ),
            index_col = "Id")

x = ames_data[ames_data.columns[ames_data.columns!="SalePrice"]]
y = ames_data["SalePrice"]

# x.shape

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


def replace_never_seen_values(train, test):
    # test.loc[~test.x.isin(train.x.unique()), test.x.name] = np.nan
    for x in test:
        test.loc[~test[x].isin(train[x].unique()), test[x].name] = test.loc[:, x].mode().values[0]
        # print(test.loc[:, x].mode())

def print_never_seen_values(train, test):
    for x in test:
        # print(test.loc[:, x].mode())
        print(x, test.loc[~test[x].isin(train[x].unique()), test[x].name])


# x_train.shape
# x_test.shape
# y_train.shape
# y_test.shape

x_train_num = VariableSelector(variable_type = "numeric").fit_transform(x_train)
x_train_cat = VariableSelector(variable_type = "categorical").fit_transform(x_train)
x_test_num = VariableSelector(variable_type = "numeric").fit_transform(x_test)
x_test_cat = VariableSelector(variable_type = "categorical").fit_transform(x_test)

x_train_num.shape
x_train_cat.shape
x_test_num.shape
x_test_cat.shape

# x_train_cat
# x_test_cat


# x_train_cat.isnull().sum().sum()

np.nan

num_imputer = Imputer(strategy = "median").fit(x_train_num)
x_train_num = num_imputer.transform(x_train_num)
x_test_num = num_imputer.transform(x_test_num)

x_train_cat = x_train_cat.fillna("None")
x_test_cat = x_test_cat.fillna("None")

# label_binarizer = LabelEncoder()
# label_binarizer = MultiLabelBinarizer()

# LabelEncoder().fit_transform(x_train_cat)



label_encoders = defaultdict(LabelEncoder)
# dir(label_encoders)

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
x_train_num = scaler.fit_transform(x_train_num)
x_test_num = scaler.transform(x_test_num)

# [label_encoders[i].get_params() for i in label_encoders]
# onehot_encoders = defaultdict(OneHotEncoder)
# defaultdict([1, 2, 3])



replace_never_seen_values(x_train_cat, x_test_cat)
# print_never_seen_values(x_train_cat, x_test_cat)

onehot_encoder = OneHotEncoder(categorical_features = "all", handle_unknown = "unknown")

x_train_label_encode = x_train_cat.apply(lambda x: label_encoders[x.name].fit_transform(x))
x_train_ready_cat = onehot_encoder.fit_transform(x_train_label_encode)
x_test_label_encode = x_test_cat.apply(lambda x: label_encoders[x.name].transform(x.astype(str)))
x_test_ready_cat = onehot_encoder.transform(x_test_label_encode)



x_train = np.hstack([x_train_num, x_train_ready_cat.toarray()])
x_test = np.hstack([x_test_num, x_test_ready_cat.toarray()])

pd.DataFrame(x_train).to_csv(os.path.join(
                            os.path.dirname(
                                    os.path.abspath("__file__")
                            ),'amnes/data/working/train_x.csv'
                        ), index = False)
pd.DataFrame(x_test).to_csv(os.path.join(
                            os.path.dirname(
                                    os.path.abspath("__file__")
                            ),'amnes/data/working/test_x.csv'
                        ), index = False)

pd.DataFrame(y_train).to_csv(os.path.join(
                            os.path.dirname(
                                    os.path.abspath("__file__")
                            ),'amnes/data/working/train_y.csv'
                        ))

pd.DataFrame(y_test).to_csv(os.path.join(
                            os.path.dirname(
                                    os.path.abspath("__file__")
                            ),'amnes/data/working/test_y.csv'
                        ))



x_train_num.shape
x_train_ready_cat.toarray().shape
x_train.shape
y_train.shape

x_test_num.shape
x_test_ready_cat.toarray().shape
x_test.shape
y_test.shape




def rmsle(y_pred, y_test) :
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))

rmsle_score = make_scorer(rmsle, greater_is_better=False)


# randint(1, 2)
# # max_features, things that count variables or splits
# randint.rvs(0, 20, size=1000)
# # C (svc) - 2, 10
# # gamma (svc) - 0.1, 1 (0.000001 to 0.9 gets limits)
# uniform.rvs(0.01, 0.9, size=1000)
#
# alpha.rvs(100, size = 100)


# full_pipeline = Pipeline([
#         ('reg', Ridge()),
#         ])

# full_pipeline = Pipeline([
#         ('reg', ElasticNet()),
#         ])

# full_pipeline = Pipeline([
#         ('reg', BayesianRidge()),
#         ])
#
full_pipeline = Pipeline([
        # ('feature_selection', SelectKBest(f_regression)),
        ('reg', BayesianRidge()),
        ])
#

pg = {
    # 'reg__C' : np.logspace(-2, 10, 50),
    # 'reg__gamma' : np.logspace(-9, 3, 50),

    'reg__alpha_1' : np.logspace(-10, -2, 100).tolist(),
    'reg__alpha_2' : np.logspace(-10, -2, 100).tolist(),
    'reg__lambda_1' : np.logspace(-10, -2, 100).tolist(),
    'reg__lambda_2' : np.logspace(-10, -2, 100).tolist(),
    'reg__fit_intercept' : [True, False],
    'reg__normalize' : [True, False],
    'reg__compute_score' : [True, False],
    'reg__tol' : uniform(0.00001, 1),
    }

# pg = {
#     'reg__alpha' : np.logspace(-10, -2, 100).tolist(),
#     'reg__l1_ratio' : uniform(0.00001, 1),
#     }

# pg = {
#     'reg__n_estimators' : randint(1, 100),
#     'reg__max_features' : randint(1, 500),
#     'reg__max_depth' : randint(1, 100),
#     'reg__algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
#     'reg__kernel' : ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
#     'reg__lambda_2' : np.logspace(-10, -2, 20).tolist(),
#     }


# pg = { # trees
#         'reg__criterion' : ['mse', 'friedman_mse', 'mae'],
    # 'reg__splitter' : ['best', 'random'],
    # 'reg__max_depth' : randint(2, 100),
    # 'reg__min_samples_split' : randint(2, 100),
    # 'reg__max_features' : randint(1, 300),
#     }

# pg = {
#     # 'feature_selection__k': list(range(2, 100, 5)),
#     # 'reg': [LinearRegression(), Lasso(), Ridge()],
#     # 'reg__alpha' : lomax(100000),
#     # 'poli__degree' : randint(1, 3),
#     'reg__fit_intercept' : [True, False],
#     'reg__normalize' : [True, False],
#     }




grid = RandomizedSearchCV(full_pipeline, param_distributions=pg,
                                        cv=10,
                                        scoring = "r2",
                                        # scoring = "r2",
                                        n_iter = 500)


# x_train.shape
poli = PolynomialFeatures(degree = 2)
# poli = PolynomialFeatures(interaction_only = True)
x_train = poli.fit_transform(x_train)
# x_train.shape
# x_train.shape

pca_transformer = PCA(n_components=1000)
pca_transformer.fit(x_train)
pca_transformer.explained_variance_ratio_.sum()
x_train = pca_transformer.transform(x_train)

# np.e ** np.log(y_train)

scaler_y = StandardScaler(copy=True, with_mean=True, with_std=True)
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
# y_train.values().reshape(-1, 1)


# y_train.values.reshape(-1, 1).shape
# x_test_num = scaler_y.transform(x_test_num)
# y_train = np.log(y_train)



# x_train



grid.fit(x_train, y_train.ravel())

pd.DataFrame(grid.grid_scores_).sort_values("mean_validation_score")


x_test = poli.transform(x_test)
x_test = pca_transformer.transform(x_test)
# pd.DataFrame(grid.grid_scores_)
# pd.DataFrame(grid.grid_scores_).sort_values(by = )
pd.DataFrame([np.e ** grid.predict(x_test), scaler_y.transform(y_test.values.reshape(-1, 1))])
mean_squared_error(y_test, np.e ** grid.predict(x_test))
# 2004,805,126
# 716,852,668
# 759,107,470
# 2057,570,962
# 1260,684,066
# 1689,874,386
# 1608,326,518
# 5405,998,897
# 715,551,980
# 778,085,588
# 804,713,150
# 938,380,884
# 793,699,783
# 4075,843,681
# 905,276,688
# 947,649,385
import matplotlib.pyplot as plt
plt.scatter(grid.predict(x_test), y_test)
# # plt.hist(alpha.rvs(1000, size = 100))
# # plt.hist(uniform.rvs(1e-20, 1e-3, size=1000))
# gamma.rvs(0.1, size = 1000)
from scipy.stats import lognorm

# plt.hist([grid.predict(x_test), y_test])
# plt.hist(gamma.rvs(10, size = 1000))
# plt.hist(gamma.rvs(100, size = 1000))
# plt.hist(gamma.rvs(1000, size = 1000))
# plt.hist(gamma.rvs(10000, size = 1000))
# # betaprime.rvs(12, 100000, size=1000)
#
lomax.rvs(100000, size = 10000)
uniform.rvs(0.00001, 0.1, size = 1000)
lomax.rvs(2, size = 100)
#
#
test_data = pd.read_csv(
                os.path.join(
                        os.path.dirname(
                                os.path.abspath("__file__")
                        ),'amnes/data/test.csv'
                    ),
            index_col = "Id")
#
#
test_data_num = VariableSelector(variable_type = "numeric").fit_transform(test_data)
test_data_cat = VariableSelector(variable_type = "categorical").fit_transform(test_data)

test_data_num.shape
test_data_cat.shape

test_data_num = num_imputer.transform(test_data_num)
test_data_cat = test_data_cat.fillna("None")

test_data_num = scaler.transform(test_data_num)

replace_never_seen_values(x_train_cat, test_data_cat)


test_data_label_encode = test_data_cat.apply(lambda x: label_encoders[x.name].transform(x.astype(str)))
test_data_ready_cat = onehot_encoder.transform(test_data_label_encode)

#
test_data_ready = np.hstack([test_data_num, test_data_ready_cat.toarray()])
pd.DataFrame(test_data_ready).to_csv(os.path.join(
                            os.path.dirname(
                                    os.path.abspath("__file__")
                            ),'amnes/data/working/target_x.csv'
                        ))
test_data_ready = poli.transform(test_data_ready)
test_data_ready = pca_transformer.transform(test_data_ready)
# test_data_ready


# results = pd.DataFrame(np.e ** grid.predict(test_data_ready)).set_index(test_data.index)
results = pd.DataFrame(scaler_y.inverse_transform(grid.predict(test_data_ready))).set_index(test_data.index)
# results = results)

results.rename(columns = {0: "SalePrice"}).to_csv(
                                            os.path.join(
                                                    os.path.dirname(
                                                            os.path.abspath("__file__")
                                                    ),'amnes/data/submission8.csv'
                                                ))

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

from sklearn.preprocessing import RobustScaler

ames_data = pd.read_csv(
                os.path.join(
                        os.path.dirname(
                                os.path.abspath("__file__")
                        ),'ames/data/train.csv'
                    ),
            index_col = "Id")

x = ames_data[ames_data.columns[ames_data.columns!="SalePrice"]]
y = ames_data["SalePrice"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train_num = VariableSelector(variable_type = "numeric").fit_transform(x_train)
x_train_cat = VariableSelector(variable_type = "categorical").fit_transform(x_train)
x_test_num = VariableSelector(variable_type = "numeric").fit_transform(x_test)
x_test_cat = VariableSelector(variable_type = "categorical").fit_transform(x_test)


num_imputer = Imputer(strategy = "median").fit(x_train_num)
x_train_num = num_imputer.transform(x_train_num)
x_test_num = num_imputer.transform(x_test_num)

x_train_cat = x_train_cat.fillna("None")
x_test_cat = x_test_cat.fillna("None")

label_encoders = defaultdict(LabelEncoder)
# dir(label_encoders)

scaler = RobustScaler()
x_train_num = scaler.fit_transform(x_train_num)
x_test_num = scaler.transform(x_test_num)


def replace_never_seen_values(train, test):
    # test.loc[~test.x.isin(train.x.unique()), test.x.name] = np.nan
    for x in test:
        test.loc[~test[x].isin(train[x].unique()), test[x].name] = test.loc[:, x].mode().values[0]

replace_never_seen_values(x_train_cat, x_test_cat)


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
                            ),'ames/data/working/robust_train_x.csv'
                        ), index = False)
pd.DataFrame(x_test).to_csv(os.path.join(
                            os.path.dirname(
                                    os.path.abspath("__file__")
                            ),'ames/data/working/robust_test_x.csv'
                        ), index = False)

pd.DataFrame(y_train).to_csv(os.path.join(
                            os.path.dirname(
                                    os.path.abspath("__file__")
                            ),'ames/data/working/robust_train_y.csv'
                        ))

pd.DataFrame(y_test).to_csv(os.path.join(
                            os.path.dirname(
                                    os.path.abspath("__file__")
                            ),'ames/data/working/robust_test_y.csv'
                        ))


test_data = pd.read_csv(
                os.path.join(
                        os.path.dirname(
                                os.path.abspath("__file__")
                        ),'ames/data/test.csv'
                    ),
            index_col = "Id")


test_data_num = VariableSelector(variable_type = "numeric").fit_transform(test_data)
test_data_cat = VariableSelector(variable_type = "categorical").fit_transform(test_data)


test_data_num = num_imputer.transform(test_data_num)
test_data_cat = test_data_cat.fillna("None")

test_data_num = scaler.transform(test_data_num)

replace_never_seen_values(x_train_cat, test_data_cat)


test_data_label_encode = test_data_cat.apply(lambda x: label_encoders[x.name].transform(x.astype(str)))
test_data_ready_cat = onehot_encoder.transform(test_data_label_encode)

#
test_data_ready = np.hstack([test_data_num, test_data_ready_cat.toarray()])
pd.DataFrame(test_data_ready).set_index(test_data.index).to_csv(os.path.join(
                            os.path.dirname(
                                    os.path.abspath("__file__")
                            ),'ames/data/working/robust_target_x.csv'
                        ))

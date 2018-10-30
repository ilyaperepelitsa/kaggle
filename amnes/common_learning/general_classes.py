from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer


from sklearn.preprocessing import Imputer
from pandas import DataFrame
import pandas as pd
import numpy as np


###
### WE KEEP ALL THE SKLEARN CUSTOM CLASSES HERE

########## SELECTORS
# class VariableSelector(BaseEstimator, TransformerMixin):
class VariableSelector(TransformerMixin):
    def __init__(self, variable_type):
        """
        variable_type: accepts three values
                        categorical - returns  columns of dtype object
                        numeric - returns  columns of dtypes float64/int64/uint8
                        already_encoded - returns  columns with unique values [0,1]
        already_encoded - dropped from both numeric and categorical columns
            reason for exlusion - they are assumed to be preprocessed categorical

        already_encoded filter (dropnas) - for each column the NA's are dropped
            to make sure that only existing values are taken into consideration.
            Neither used for imputing nor dropping NAs from dataframes that are
            to be returned.
        """

        self.variable_type = variable_type
        self.columns = []
        # self.target = target

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        return dict(variable_type=self.variable_type, columns = self.columns)
        # return dict(variable_type=self.variable_type)

    # def get_feature_names(self):
    #     return self.columns.tolist()

    def transform(self, X):
        if self.variable_type == "categorical":
            already_encoded_labels = []
            for i in X.columns:
                if sorted(X[i].dropna().unique().tolist()) == [0, 1]:
                    already_encoded_labels.append(X[i].name)

            for label in already_encoded_labels:
                if label not in X.select_dtypes(include = ["object"]).columns:
                    already_encoded_labels.remove(label)

            # if self.target in X.select_dtypes(include = ["object"]).columns:
            #     return X.select_dtypes(include = ["object"]).drop(already_encoded_labels, axis=1).drop(self.target, axis=1)
            # else:
            #     return X.select_dtypes(include = ["object"]).drop(already_encoded_labels, axis=1)
            self.columns = X.select_dtypes(include = ["object"]).drop(already_encoded_labels, axis=1).columns.tolist()
            return X.select_dtypes(include = ["object"]).drop(already_encoded_labels, axis=1)

        elif self.variable_type == "numeric":

            already_encoded_labels = []
            for i in X.columns:
                if sorted(X[i].dropna().unique().tolist()) == [0, 1]:
                    already_encoded_labels.append(X[i].name)


            for label in already_encoded_labels:
                if label not in X.select_dtypes(include = ["float64", "int64", "uint8"]).columns:
                    already_encoded_labels.remove(label)

            # if self.target in X.select_dtypes(include = ["float64", "int64", "uint8"]).columns:
            #     return X.select_dtypes(include = ["float64", "int64", "uint8"]).drop(already_encoded_labels, axis=1).drop(self.target, axis=1)
            # else:
            #     return X.select_dtypes(include = ["float64", "int64", "uint8"]).drop(already_encoded_labels, axis=1)
            self.columns = X.select_dtypes(include = ["float64", "int64", "uint8"]).drop(already_encoded_labels, axis=1).columns.tolist()
            return X.select_dtypes(include = ["float64", "int64", "uint8"]).drop(already_encoded_labels, axis=1)

        elif self.variable_type == "already_encoded":
            already_encoded_labels = []
            for i in X.columns:
                if sorted(X[i].dropna().unique().tolist()) == [0, 1]:
                    already_encoded_labels.append(X[i].name)

            # if self.target in X[already_encoded_labels].columns:
            #     return X[already_encoded_labels].drop(self.target, axis=1)
            # else:
            #     return X[already_encoded_labels]
            self.columns = X[already_encoded_labels].columns.tolist()
            return X[already_encoded_labels]

# class ColumnSelector(BaseEstimator, TransformerMixin):
class ColumnSelector(TransformerMixin):
    def __init__(self, column_name_choice):
        # """
        # variable_type: accepts three values
        #                 categorical - returns  columns of dtype object
        #                 numeric - returns  columns of dtypes float64/int64/uint8
        #                 already_encoded - returns  columns with unique values [0,1]
        # already_encoded - dropped from both numeric and categorical columns
        #     reason for exlusion - they are assumed to be preprocessed categorical
        #
        # already_encoded filter (dropnas) - for each column the NA's are dropped
        #     to make sure that only existing values are taken into consideration.
        #     Neither used for imputing nor dropping NAs from dataframes that are
        #     to be returned.
        # """
        self.column_name_choice = column_name_choice

    def get_params(self, deep=True):
        return dict(column_name_choice=self.column_name_choice)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.column_name_choice in X.columns:
            return np.asmatrix(X[self.column_name_choice]).transpose()


########## TRANSFORMERS
# class NoneImputer(BaseEstimator, TransformerMixin):
class NoneImputer(TransformerMixin):

    def __init__(self, impute_string):
        """
        This imputer populates missing values with a string "None".
            "None" is chosen to replace features that are actually missing
            for a given observation rather than "not recorded". Reflects the way
            R treated Null and NA values as two different types of missing values.
        """

        self.impute_string = impute_string

        pass
    def fit(self, X, y=None):
        self.fill = self.impute_string
        return self

    def get_params(self, deep=True):
        return dict(impute_string=self.impute_string)
    # def get_feature_names(self):
    #     return self.columns.tolist()

    def transform(self, X, y=None):
        X[pd.isnull(np.asarray(X))] = self.fill
        return np.asarray(X)
        # return X.fillna(self.fill).values

class CustomBinarizer(MultiLabelBinarizer):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y = None):
        return self.transform(X)
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

class LabelBinarizerPipelineFriendly(MultiLabelBinarizer):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(LabelBinarizerPipelineFriendly, self).fit(X)
    def transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)
    def fit_transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)
# #
# class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
#     def __init__(self, sparse_output=False):
#         self.sparse_output = sparse_output
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X, y=None):
#         enc = LabelBinarizer(sparse_output=self.sparse_output)
#         return enc.fit_transform(X)
#
# class LabelBinarizer_new(TransformerMixin, BaseEstimator):
#     def fit(self, X, y = 0):
#         self.encoder = None
#         return self
#     def transform(self, X, y = 0):
#         if(self.encoder is None):
#             print("Initializing encoder")
#             self.encoder = LabelBinarizer();
#             result = self.encoder.fit_transform(X)
#         else:
#             result = self.encoder.transform(X)
#         return result;
#
# class LabelBinarizer_new_2(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         X = self.prep(X)
#         unique_vals = []
#         for column in X.T:
#             unique_vals.append(np.unique(column))
#         self.unique_vals = unique_vals
#     def transform(self, X, y=None):
#         X = self.prep(X)
#         unique_vals = self.unique_vals
#         new_columns = []
#         for i, column in enumerate(X.T):
#             num_uniq_vals = len(unique_vals[i])
#             encoder_ring = dict(zip(unique_vals[i], range(len(unique_vals[i]))))
#             f = lambda val: encoder_ring[val]
#             f = np.vectorize(f, otypes=[np.int])
#             new_column = np.array([f(column)])
#             if num_uniq_vals <= 2:
#                 new_columns.append(new_column)
#             else:
#                 one_hots = np.zeros([num_uniq_vals, len(column)], np.int)
#                 one_hots[new_column, range(len(column))]=1
#                 new_columns.append(one_hots)
#         new_columns = np.concatenate(new_columns, axis=0).T
#         return new_columns
#
#     def fit_transform(self, X, y=None):
#         self.fit(X)
#         return self.transform(X)
#
#     @staticmethod
#     def prep(X):
#         shape = X.shape
#         if len(shape) == 1:
#             X = X.values.reshape(shape[0], 1)
#         return X

# class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
#     """Perform one-hot encoding to categorical features."""
#     def __init__(self, cat_features):
#         self.cat_features = cat_features
#
#     def fit(self, X_cat, y=None):
#         return self
#
#     def transform(self, X_cat):
#         X_cat_df = pd.DataFrame(X_cat, columns=self.cat_features)
#         X_onehot_df = pd.get_dummies(X_cat_df, columns=self.cat_features)
#         return X_onehot_df.values

# class ArrayToDF(BaseEstimator, TransformerMixin):
class ArrayToDF(TransformerMixin):
    def __init__(self):
        # """
        # variable_type: accepts three values
        #                 categorical - returns  columns of dtype object
        #                 numeric - returns  columns of dtypes float64/int64/uint8
        #                 already_encoded - returns  columns with unique values [0,1]
        # already_encoded - dropped from both numeric and categorical columns
        #     reason for exlusion - they are assumed to be preprocessed categorical
        #
        # already_encoded filter (dropnas) - for each column the NA's are dropped
        #     to make sure that only existing values are taken into consideration.
        #     Neither used for imputing nor dropping NAs from dataframes that are
        #     to be returned.
        # """
        # self.column_name = column_name
        pass
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X)


########## REGRESSORS
########## CLASSIFIERS
########## CLUSTER







#### INSPIRATION

# https://stackoverflow.com/questions/42846345/sklearn-categorical-imputer
# class SeriesImputer(TransformerMixin):
#
#     def __init__(self):
#         """Impute missing values.
#
#         If the Series is of dtype Object, then impute with the most frequent object.
#         If the Series is not of dtype Object, then impute with the mean.
#
#         """
#     def fit(self, X, y=None):
#         if   X.dtype == numpy.dtype('O'): self.fill = X.value_counts().index[0]
#         else                            : self.fill = X.mean()
#         return self
#
#     def transform(self, X, y=None):
#         return X.fillna(self.fill)

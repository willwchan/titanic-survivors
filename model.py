import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

titanic_data = pd.read_csv('train.csv')

y = titanic_data.Survived
X = titanic_data.drop(['Survived'], axis=1)

#split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)

#drop columns with missing values
cols_to_drop = [col for col in X_train.columns if X_train[col].isnull().any()]

X_train = X_train.drop(cols_to_drop, axis=1)
X_valid = X_valid.drop(cols_to_drop, axis=1)

#drop columns with categorical values (mae: 0.3584)
# X_train = X_train.select_dtypes(exclude=['object'])
# X_valid = X_valid.select_dtypes(exclude=['object'])

#one hot encode categorical values (mae: 0.2587)
object_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']

#apply oh encoder to each categorical data column
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

#oh encoder removes columns so we have to put them back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

#then remove the categorical columns bc we replace them with oh encoding
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

#lastly add the oh encoded columns to the numerical features
X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

#train model and predict
# for estimators in [600,650,700,750,800,850,900]:
my_model = XGBRegressor(n_estimators = 2000, learning_rate = 0.05)
my_model.fit(X_train, y_train) #early_stopping_rounds = 5, eval_set = [(X_valid, y_valid)], verbose = False)
predictions = my_model.predict(X_valid)
print("MAE: "+str(mean_absolute_error(predictions, y_valid)))

#predict on test data
test_data = pd.read_csv('test.csv')
pred_test_data = my_model.predict(test_data)

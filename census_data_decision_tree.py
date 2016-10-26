import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, tree 

#Definind columns and categorial columns
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country", "income_bracket"]

#reading data and adding column names to the data.
df_train = pd.read_csv('adult.data.txt', names = COLUMNS) 
df_test = pd.read_csv('adult.test.txt', names = COLUMNS, skiprows =1) 

#filling Nan's with -999999
df_train.fillna(-99999)
df_test.fillna(-99999)

#Encode categorial columns with numbers 0,1,2,etc.
le = preprocessing.LabelEncoder()

for col in CATEGORICAL_COLUMNS:
	if df_train[col].dtypes == 'object':
		data = df_train[col].append(df_test[col])
		le.fit(data.values)
		df_train[col] = le.transform(df_train[col])
		df_test[col] = le.transform(df_test[col])

#Define train and test data as np array.
X_test = np.array(df_train.drop(['income_bracket'], 1))
y_test = np.array(df_train['income_bracket'])
X = np.array(df_train.drop(['income_bracket'], 1))
y = np.array(df_train['income_bracket'])

#Run the Decisiontree classifier on train data and test the accuracy on test data.
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)
accuracy = clf.score(X_test, y_test)
print(accuracy)
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"D:\project\adult\venv\adult.csv")

# Handle missing values
df.replace("?", np.nan, inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# Discretize marital status
df.replace(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 
            'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'],
           ['divorced', 'married', 'married', 'married', 
            'not married', 'not married', 'not married'], inplace=True)

# Label encoding
category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                'relationship', 'gender', 'native-country', 'income']
label_encoder = preprocessing.LabelEncoder()
mapping_dict = {}

for col in category_col:
    df[col] = label_encoder.fit_transform(df[col])
    mapping_dict[col] = dict(enumerate(label_encoder.classes_))

print(mapping_dict)

# Drop unwanted columns
df.drop(['fnlwgt', 'educational-num'], axis=1, inplace=True)

# Split features and target
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# Train Decision Tree
dt_clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=5, min_samples_leaf=5)
dt_clf_gini.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as model_file:
    pickle.dump(dt_clf_gini, model_file)

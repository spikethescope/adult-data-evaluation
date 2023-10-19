# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
# Load the dataset
# Define the data types of the columns in the CSV file
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None)

# Define the column names
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df.replace("?", "unknown", inplace=True)
print(df)
# Encode categorical features (Outlook, Temp, Humid, Wind) into numerical values
label_encoders = {}
for column in df.columns:
    le = preprocessing.LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

print(df)


# Convert the target variable to binary
#df['income'] = df['income'].apply(lambda x: 0 if x == ' <=50K' else 1)
# Get a list of all the numeric columns in the dataset

X = df.drop('income', axis=1)
y = df['income']
print(X)
print(y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RANDOM FOREST CLASSIFIER


mtry = 10
rf = RandomForestClassifier(n_estimators=100,max_features=mtry, random_state=42)
rf.fit(X_train, y_train)
y_pred_RF = rf.predict(X_test)

y_pred_proba_RF = rf.predict_proba(X_test)[:, 1]
accuracy_RF = accuracy_score(y_test, y_pred_RF)
print("Accuracy with mtry={}: {:.2f}".format(mtry, accuracy_RF))

# DECISION TREE CLASSIFIER


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_DT = dt.predict(X_test)

y_pred_proba_DT = dt.predict_proba(X_test)[:, 1]
accuracy_DT = accuracy_score(y_test, y_pred_DT)
print("Accuracy with {:.2f}".format(accuracy_DT))

fpr_DT, tpr_DT, thresholds_DT = roc_curve(y_test, y_pred_proba_DT)
roc_auc_DT = auc(fpr_DT, tpr_DT)


fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_test, y_pred_proba_RF)
roc_auc_RF = auc(fpr_RF, tpr_RF)


# Plot the ROC curves
plt.figure()
plt.plot(fpr_DT, tpr_DT, label='Decision Tree Classifier' % (roc_auc_DT))
plt.plot(fpr_RF, tpr_RF, label='Random Forest Classifier' % (roc_auc_RF))


# Add labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()

from sklearn.model_selection import cross_val_score

mtry = 10
rf = RandomForestClassifier(n_estimators=100,max_features=mtry, random_state=42)
# Perform cross-validation
scores = cross_val_score(rf, X_train, y_train, cv=5)
print(scores)

# Print the mean score and standard deviation
print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Fit the model on the entire training set
rf.fit(X_train, y_train)
"""
# Get the mean score for each fold
mean_scores = scores.mean()
# Select the model with the highest mean score
best_model_idx = np.argmax(scores)

# Get the best model
best_model = rf.fit(X_train, y_train)
"""
# Evaluate the best model on the test data
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy on the test set:', accuracy)

from sklearn.model_selection import GridSearchCV

# Create a grid of hyperparameters to search over
param_grid = {
    'n_estimators': [25, 75, 100],
    'max_features': ['sqrt', 'log2']
}

# Create a GridSearchCV object
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=5
)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on the test data
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy on the test set:', accuracy)


# Plot the feature importances
features = X_train.columns
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), features[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()
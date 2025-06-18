import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,learning_curve
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
import seaborn as sns


import matplotlib.pyplot as plt

import numpy as np

data_path="../raw_data/wdbc.data"


#adding the column names 

columns=['id','diagnosis',*['feature_' + str(i) for i in range(1, 31)]]
df=pd.read_csv(data_path,header=None, names=columns)


# Dropping the 'id' column
df.drop(columns=['id'], axis=1)

# Encoding the 'diagnosis' column
df['diagnosis']=df['diagnosis'].map({'M': 1, 'B': 0})


# Splitting the dataset into features and target variable
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
print(len(X_train))
X_test = scaler.transform(X_test)
print(len(X_test))


# Training the SVM model
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)


# Making predictions on the test set
y_pred = model.predict(X_test)


parm_grid={
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf','poly']
}

# Performing Grid Search for hyperparameter tuning
grid_search = GridSearchCV(SVC(random_state=42), parm_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)


# Best parameters from grid search
best_params = grid_search.best_params_
print("Best Parameters from Grid Search:", best_params)

# Retraining the model with the best parameters
model = SVC(**best_params, random_state=42)
model.fit(X_train, y_train)

# Making predictions with the tuned model
y_pred = model.predict(X_test)  


# Evaluate on test set
y_pred = grid_search.predict(X_test)
print("Test set accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Evaluating the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot it
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")


model = SVC(**grid_search.best_params_, random_state=42)

# Get learning curve data
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=10, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

# Calculate mean and std
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training Score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label='Cross-validation Score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='green')
plt.title("Learning Curve for SVM with Tuned Hyperparameters")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("learning_curve.png")


# Gradient Boosting - Lab

## Introduction

In this lab, we'll learn how to use both Adaboost and Gradient Boosting classifiers from scikit-learn!

## Objectives

You will be able to:

- Use AdaBoost to make predictions on a dataset 
- Use Gradient Boosting to make predictions on a dataset 

## Getting Started

In this lab, we'll learn how to use boosting algorithms to make classifications on the [Pima Indians Dataset](http://ftp.ics.uci.edu/pub/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.names). You will find the data stored in the file `'pima-indians-diabetes.csv'`. Our goal is to use boosting algorithms to determine whether a person has diabetes. Let's get started!

We'll begin by importing everything we need for this lab. Run cell below:


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
```

Now, use Pandas to import the data stored in `'pima-indians-diabetes.csv'` and store it in a DataFrame. Print the first five rows to inspect the data we've imported and ensure everything loaded correctly. 


```python
# Import the data
df = None

# Print the first five rows

```

## Cleaning, exploration, and preprocessing

The target we're trying to predict is the `'Outcome'` column. A `1` denotes a patient with diabetes. 

By now, you're quite familiar with exploring and preprocessing a dataset.  

In the following cells:

* Check for missing values and deal with them as you see fit (if any exist) 
* Count the number of patients with and without diabetes in this dataset 
* Store the target column in a separate variable and remove it from the dataset
* Split the dataset into training and test sets, with a `test_size` of 0.25 and a `random_state` of 42


```python
# Check for missing values

```


```python
# Number of patients with and without diabetes

```


```python
target = None
df = None
```


```python
# Split the data into training and test sets
X_train, X_test, y_train, y_test = None
```

## Train the models

Now that we've explored the dataset, we're ready to fit some models!

In the cell below:

* Instantiate an `AdaBoostClassifier` (set the `random_state` for 42)
* Instantiate a `GradientBoostingClassifer` (set the `random_state` for 42) 


```python
# Instantiate an AdaBoostClassifier
adaboost_clf = None

# Instantiate an GradientBoostingClassifier
gbt_clf = None
```

Now, fit the training data to both the classifiers: 


```python
# Fit AdaBoostClassifier

```


```python
# Fit GradientBoostingClassifier

```

Now, let's use these models to predict labels on both the training and test sets: 


```python
# AdaBoost model predictions
adaboost_train_preds = None
adaboost_test_preds = None

# GradientBoosting model predictions
gbt_clf_train_preds = None
gbt_clf_test_preds = None
```

Now, complete the following function and use it to calculate the accuracy and f1-score for each model: 


```python
def display_acc_and_f1_score(true, preds, model_name):
    acc = None
    f1 = None
    print("Model: {}".format(model_name))
    print("Accuracy: {}".format(None))
    print("F1-Score: {}".format(None))
    
print("Training Metrics")
display_acc_and_f1_score(y_train, adaboost_train_preds, model_name='AdaBoost')
print("")
display_acc_and_f1_score(y_train, gbt_clf_train_preds, model_name='Gradient Boosted Trees')
print("")
print("Testing Metrics")
display_acc_and_f1_score(y_test, adaboost_test_preds, model_name='AdaBoost')
print("")
display_acc_and_f1_score(y_test, gbt_clf_test_preds, model_name='Gradient Boosted Trees')
```

Let's go one step further and create a confusion matrix and classification report for each. Do so in the cell below: 


```python
adaboost_confusion_matrix = None
adaboost_confusion_matrix
```


```python
gbt_confusion_matrix = None
gbt_confusion_matrix
```


```python
adaboost_classification_report = None
print(adaboost_classification_report)
```


```python
gbt_classification_report = None
print(gbt_classification_report)
```

**_Question:_** How did the models perform? Interpret the evaluation metrics above to answer this question.

Write your answer below this line:
_______________________________________________________________________________________________________________________________

 
 
As a final performance check, let's calculate the 5-fold cross-validated score for each model! 

Recall that to compute the cross-validation score, we need to pass in:

* A classifier
* All training data
* All labels
* The number of folds we want in our cross-validation score  

Since we're computing cross-validation score, we'll want to pass in the entire dataset, as well as all of the labels. 

In the cells below, compute the mean cross validation score for each model. 


```python
print('Mean Adaboost Cross-Val Score (k=5):')
print(None)
# Expected Output: 0.7631270690094218
```


```python
print('Mean GBT Cross-Val Score (k=5):')
print(None)
# Expected Output: 0.7591715474068416
```

These models didn't do poorly, but we could probably do a bit better by tuning some of the important parameters such as the **_Learning Rate_**. 

## Summary

In this lab, we learned how to use scikit-learn's implementations of popular boosting algorithms such as AdaBoost and Gradient Boosted Trees to make classification predictions on a real-world dataset!

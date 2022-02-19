#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 18:42:16 2022

@author: alishaparveen
"""
# Data Analysis and Wrangling
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action="ignore")
from sklearn.preprocessing import Normalizer
# Model Evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
# Supervised machine learning algorithms#
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Hyper Parameter Tuning and Strata Cross validation#
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# Generation of Performance report of classification problem#
from sklearn.metrics import classification_report
from statistics import mean, stdev 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

# Defining the working directory
input_path = '../Users/alishaparveen/Bbankbr/'
#Data preprocessing#
DF = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', "*.csv"))))
DF.replace("?", np.nan, inplace = True)
plt.figure(figsize=(50,16), dpi=300)
sns.heatmap(DF.isnull(), cbar=False, cmap="BuPu")
DF=DF.dropna()
DF.info()
DF.duplicated().sum()
DF=DF.drop_duplicates()
sns.heatmap(DF.isnull(), cbar=False, cmap="PiYG")

#Exploring class label#
print(DF['class'].value_counts())
print('-'* 30)
print('bankrupt: ', round(DF['class'].value_counts()[0]/len(DF) * 100,2), '% of the dataset')
print('no banruptcy: ', round(DF['class'].value_counts()[1]/len(DF) * 100,2), '% of the dataset')

# Stratified Cross Validation Splitting
X = Normalizer().fit_transform(DF.loc[:, DF.columns!='class'])
Y=DF.loc[:, 'class']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    stratify=Y,
                                                    random_state=42)

X_train = pd.DataFrame (X_train)
X_test = pd.DataFrame (X_test)
y_train = pd.DataFrame (y_train)
y_test = pd.DataFrame (y_test)
oversampled = SMOTE(random_state=0)
X_train_sm, y_train_sm = oversampled.fit_resample(X_train, y_train)
X_test_sm, y_test_sm = oversampled.fit_resample(X_test, y_test)

###############################################################################################################################
############################################################### I. Logistics regression #######################################
###############################################################################################################################
#1. Base model-LR#
LogR_classifier = LogisticRegression(C=1)
LogR_classifier.fit(X_train_sm, y_train_sm)
LogR_prediction = LogR_classifier.predict(X_test_sm)
LogR_report = classification_report(y_test_sm, LogR_prediction)
print(LogR_report)
#2. 10FOLD Cross validation on base model-LR#
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X_train_sm, y_train_sm): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    LogR_classifier.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(LogR_classifier.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
#3. Hyperparameter tuning#
penalty = ['l1', 'l2'] 
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
solver = ['liblinear', 'saga']
param_grid = dict(penalty=penalty,
                  C=C,
                  #class_weight=class_weight,
                  solver=solver)
LR_grid = GridSearchCV(estimator=LogR_classifier,
                    param_grid=param_grid,
                    scoring='accuracy',
                    verbose=1,
                    n_jobs=-1)
LR_grid_result = LR_grid.fit(X_train_sm, y_train_sm)
grid_LR_predictions = LR_grid_result.predict(X_test_sm)
grid_LR_report = classification_report(y_test_sm, grid_LR_predictions)
print(grid_LR_report)
print('Best Score: ', LR_grid_result.best_score_)
print('Best Params: ', LR_grid_result.best_params_)
#4. 10FOLD Cross validation on tunned parameter model#
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X, Y): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    LR_grid_result.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(LR_grid_result.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
#5. Performance metrics of tunned parameter model#
tn, fp, fn, tp = confusion_matrix(y_test_sm, grid_LR_predictions).ravel()
accuracy = accuracy_score(y_test_sm, grid_LR_predictions)
precision = precision_score(y_test_sm, grid_LR_predictions)
recall = recall_score(y_test_sm, grid_LR_predictions)
f1 = f1_score(y_test_sm, grid_LR_predictions)
roc_auc = roc_auc_score(y_test_sm, grid_LR_predictions)
avg_precision = average_precision_score(y_test_sm, grid_LR_predictions)
df_result = pd.DataFrame(columns=['model', 'tp', 'tn', 'fp', 'fn', 'correct', 'incorrect',
                                  'accuracy', 'precision', 'recall', 'f1', 'roc_auc','avg_pre'])
row = {'model': 'LR with SMOTE',
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'correct': tp+tn,
        'incorrect': fp+fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'avg_pre': round(avg_precision,3),       
    }
df_result = df_result.append(row, ignore_index=True)
df_result.head()
#6. ROC curve on tunned model#
avg_precision = average_precision_score(y_test_sm, grid_LR_predictions)
precision, recall, _ = precision_recall_curve(y_test_sm, grid_LR_predictions)
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Logistics Regression Precision-Recall curve: ~{0:0.4f}'.format(avg_precision))

###############################################################################################################################
############################################################### II. SVC CLASSIFIER #############################################
###############################################################################################################################
#1. Base model- SVC#
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train_sm, y_train_sm)
svc_predictions = svclassifier.predict(X_test_sm)
svc_report = classification_report(y_test_sm, svc_predictions)
print(svc_report)
#2. 10FOLD Cross validation on base model#
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X_train_sm, y_train_sm): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    svclassifier.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(svclassifier.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
#3. Hyperparameter tuning#
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
#class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
gamma = ['scale', 'auto']
param_grid = dict(C=C,
                  kernel=kernel, gamma=gamma)
svc_grid = GridSearchCV(estimator=svclassifier,
                    param_grid=param_grid,
                    scoring='accuracy',
                    verbose=1,
                    n_jobs=-1)
svc_grid_result = svc_grid.fit(X_train_sm, y_train_sm)
grid_svc_predictions = svc_grid_result.predict(X_test_sm)
grid_svc_report = classification_report(y_test_sm, grid_svc_predictions)
print(grid_svc_report)
print('Best Score: ', svc_grid_result.best_score_)
print('Best Params: ', svc_grid_result.best_params_)
#4. 10FOLD Cross validation on tunned parameter model#
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X_train_sm, y_train_sm): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    svc_grid_result.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(svc_grid_result.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
#5. Performance metrices of tunned parameter model#
tn, fp, fn, tp = confusion_matrix(y_test_sm, grid_svc_predictions).ravel()
accuracy = accuracy_score(y_test_sm, grid_svc_predictions)
precision = precision_score(y_test_sm, grid_svc_predictions)
recall = recall_score(y_test_sm, grid_svc_predictions)
f1 = f1_score(y_test_sm, grid_svc_predictions)
roc_auc = roc_auc_score(y_test_sm, grid_svc_predictions)
avg_precision = average_precision_score(y_test_sm, grid_svc_predictions)
row = {'model': 'SVM with SMOTE',
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'correct': tp+tn,
        'incorrect': fp+fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'avg_pre': round(avg_precision,3),       
    }
df_result = df_result.append(row, ignore_index=True)
#6. ROC curve of tunned parameter model#
avg_precision = average_precision_score(y_test_sm, grid_svc_predictions)
precision, recall, _ = precision_recall_curve(y_test_sm, grid_svc_predictions)
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('SVM Precision-Recall curve: ~{0:0.4f}'.format(avg_precision))

###############################################################################################################################
############################################################### III. Random Forest ##############################################
###############################################################################################################################
#1. Base model#
RFClassifier= RandomForestClassifier(n_estimators= 20, max_depth=3, random_state=0)
RFClassifier.fit(X_train_sm,y_train_sm)
RFC_predictions = RFClassifier.predict(X_test_sm)
RFC_report = classification_report(y_test_sm, RFC_predictions)
print(RFC_report)
#2. 10FOLD Cross validation on base model#
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X_train_sm, y_train_sm): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    RFClassifier.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(RFClassifier.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
#3. Hyperparameter tuning#
param_grid = { 
    'n_estimators': [100, 200, 500, 600],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
RF_grid = GridSearchCV(estimator=RFClassifier,
                    param_grid=param_grid,
                    scoring='accuracy',
                    verbose=1,
                    n_jobs=-1)
grid_RF_result = RF_grid.fit(X_train_sm, y_train_sm)
grid_RF_predictions = grid_RF_result.predict(X_test_sm)
grid_RF_report = classification_report(y_test_sm, grid_RF_predictions)
print(grid_RF_report)
print('Best Score: ', grid_RF_result.best_score_)
print('Best Params: ', grid_RF_result.best_params_)
#4. 10FOLD Cross validation on tunned parameter model#
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X_train_sm, y_train_sm): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    grid_RF_result.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(grid_RF_result.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
##5. Performance metrices on tunned parameter model#
tn, fp, fn, tp = confusion_matrix(y_test_sm, grid_RF_predictions).ravel()
accuracy = accuracy_score(y_test_sm, grid_RF_predictions)
precision = precision_score(y_test_sm, grid_RF_predictions)
recall = recall_score(y_test_sm, grid_RF_predictions)
f1 = f1_score(y_test_sm, grid_RF_predictions)
roc_auc = roc_auc_score(y_test_sm, grid_RF_predictions)
avg_precision = average_precision_score(y_test_sm, grid_RF_predictions)
row = {'model': 'RF with SMOTE',
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'correct': tp+tn,
        'incorrect': fp+fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'avg_pre': round(avg_precision,3),       
    }
df_result = df_result.append(row, ignore_index=True)
#6. Precision-recall Curve on tunned parameter model#
avg_precision = average_precision_score(y_test_sm, grid_RF_predictions)
precision, recall, _ = precision_recall_curve(y_test_sm, grid_RF_predictions)
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Random Forest Precision-Recall curve: ~{0:0.4f}'.format(avg_precision))

# =============================================================================
# #RESULTS - Multiple ROC curve after smote on tunned parameter
# =============================================================================
from sklearn.metrics import roc_curve
LR_fpr, LR_tpr, LR_thresold = roc_curve(y_test_sm, grid_LR_predictions)
SVM_fpr, SVM_tpr, SVM_threshold = roc_curve(y_test_sm, grid_svc_predictions)
RF_fpr, RF_tpr, RF_thresold = roc_curve(y_test_sm, grid_RF_predictions)

def graph_roc_curve_multiple(LR_fpr, LR_tpr,SVM_fpr,SVM_tpr,RF_fpr, RF_tpr):
    plt.figure(figsize=(20,8))
    plt.title('ROC Curve', fontsize=14)
    plt.plot(LR_fpr, LR_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_test_sm, grid_LR_predictions)))
    plt.plot(SVM_fpr, SVM_tpr, label='SVM Classifier Score: {:.4f}'.format(roc_auc_score(y_test_sm, grid_svc_predictions)))
    plt.plot(RF_fpr, RF_tpr, label='Random Forest Classifier Score: {:.4f}'.format(roc_auc_score(y_test_sm, grid_RF_predictions)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()
    
graph_roc_curve_multiple(LR_fpr, LR_tpr, SVM_fpr, SVM_tpr,RF_fpr, RF_tpr)
plt.show()


# =============================================================================
# #Prediction on 4th year
# =============================================================================
fourth_year= pd.read_csv('4year.csv')
fourth_year.replace("?", np.nan, inplace = True)
plt.figure(figsize=(50,16), dpi=300)
fourth_year=fourth_year.dropna()
fourth_year.duplicated().sum()
fourth_year=fourth_year.drop_duplicates()
x = Normalizer().fit_transform(fourth_year.loc[:, fourth_year.columns!='class'])
y=fourth_year.loc[:, 'class']
x_4_train, x_4_test, Y_4_train, Y_4_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=42)
x_4_train = pd.DataFrame(x_4_train)
x_4_test = pd.DataFrame(x_4_test)
Y_4_train = pd.DataFrame(Y_4_train)
Y_4_test = pd.DataFrame(Y_4_test)
#grid_RF_result = RF_grid.fit(x_4_train, Y_4_train)
grid_RF_4_predictions = grid_RF_result.predict(x_4_train)
df_4_result = pd.DataFrame(columns=['model', 'tp', 'tn', 'fp', 'fn', 'correct', 'incorrect',
                                  'accuracy', 'precision', 'recall', 'f1', 'roc_auc','avg_pre'])
tn, fp, fn, tp = confusion_matrix(Y_4_test, grid_RF_4_predictions).ravel()
accuracy = accuracy_score(Y_4_test, grid_RF_4_predictions)
precision = precision_score(Y_4_test, grid_RF_4_predictions)
recall = recall_score(Y_4_test, grid_RF_4_predictions)
f1 = f1_score(Y_4_test, grid_RF_4_predictions)
roc_auc = roc_auc_score(Y_4_test, grid_RF_4_predictions)
avg_precision = average_precision_score(Y_4_test, grid_RF_4_predictions)
row = {'model': 'RF with SMOTE',
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'correct': tp+tn,
        'incorrect': fp+fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'avg_pre': round(avg_precision,3),       
    }
df_4_result = df_result.append(row, ignore_index=True)
















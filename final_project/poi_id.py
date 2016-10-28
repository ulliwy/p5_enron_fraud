#!/usr/bin/python

import sys
import pickle
import numpy as np
from time import time
from sklearn import model_selection
from sklearn import tree
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from clean_data import clean_data, find_missing_values_ratio, draw_scatter
from select_features import create_barchart, calc_recall_pres
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#selected:
features_list = ['poi', 'bonus', 'expenses', 'exercised_stock_options', 'from_messages_ratio']

#all:
_features_list = ['poi', 'salary', 'to_messages', 'deferral_payments',
                  'total_payments', 'exercised_stock_options', 'bonus',
                  'restricted_stock', 'restricted_stock_deferred',
                  'expenses', 'loan_advances', 'from_messages', 'other',
                  'from_this_person_to_poi', 'director_fees', 'deferred_income',
                  'long_term_incentive', 'from_poi_to_this_person',
                  'to_messages_ratio', 'from_messages_ratio']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

missing_values_ratio = find_missing_values_ratio(data_dict)

#clean
for item in _features_list:
    data_dict = clean_data(data_dict, item)

draw_scatter(data_dict, ['salary', 'bonus'], 'raw_salary_bonus.png')

### Task 2: Remove outliers
del(data_dict['TOTAL'])
del(data_dict['THE TRAVEL AGENCY IN THE PARK'])

draw_scatter(data_dict, ['salary', 'bonus'], 'cleaned_salary_bonus.png')

poi_num = 0
non_poi_num = 0

for i in data_dict.keys():
    if data_dict[i]['poi'] == 1:
        poi_num += 1
    elif data_dict[i]['poi'] == 0:
        non_poi_num += 1

print 'DATA SUMMARY'
print 'Number of records in dataset: ', len(data_dict)
print 'Number of POI: ', poi_num
print 'Number of non-POI: ', non_poi_num
print 'Number of features used: ', len(features_list)
print 'Features with most missing values (> 80%): '

for k in missing_values_ratio.keys():
    if missing_values_ratio[k] >= 0.8:
        print '\t %s %0.2f' % (k, missing_values_ratio[k] * 100)
print '--------------------------------------'


### Task 3: Create new feature(s)
my_dataset = data_dict

for i in data_dict.keys():
    if data_dict[i]['to_messages'] == 0:
        my_dataset[i]['to_messages_ratio'] = 0
    else:
        my_dataset[i]['to_messages_ratio'] = \
            round(float(data_dict[i]['from_poi_to_this_person']) / float(data_dict[i]['to_messages']), 3)

    if data_dict[i]['from_messages'] == 0:
        my_dataset[i]['from_messages_ratio'] = 0
    else:
        my_dataset[i]['from_messages_ratio'] = \
            round(float(data_dict[i]['from_this_person_to_poi']) / float(data_dict[i]['from_messages']), 3)

### Extract features and labels from dataset for local testing

#selected:
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
# all:
_data = featureFormat(my_dataset, _features_list, sort_keys=True)
_labels, _features = targetFeatureSplit(_data)

#feature selection using KBest:
selection = SelectKBest(f_classif, k=1)
selection.fit(_features, _labels)

print 'FEATURE SELECTION'
print 'SelectKBest'
print 'Top p-values: '
for i in range(0, len(selection.scores_)):
    pval = -np.log10(selection.pvalues_[i])
    if pval > 3.0:
        print '\t', _features_list[i+1], ':', selection.scores_[i]

print 'Creating plots for SelectKBest...'
sorted_feature_list = create_barchart(_features_list[1:], selection.scores_,
                                      'selectkbest_barchart.png', 'SelectKBest Features Scores')
calc_recall_pres(sorted_feature_list, my_dataset, 'selectkbest_prec_recall.png')
print 'Done!'
print

print 'RFE and DecisionTreeClassifier'
estimator = tree.DecisionTreeClassifier()

selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(_features, _labels)

print 'Top ranked features: '
rank = selector.ranking_
for i in range(0, len(rank)):
    if rank[i] < 2:
        print '\t', _features_list[i+1], ':', rank[i]
print

print 'Creating plots for RFE...'
sorted_feature_list2 = create_barchart(_features_list[1:], list(selector.ranking_),
                                       'rfe_barchart.png', 'RFE Features Scores')
sorted_feature_list2.reverse()
calc_recall_pres(sorted_feature_list2, my_dataset, 'rfe_prec_recall.png')
print 'Done!'


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

features_train, features_test, labels_train, labels_test = \
    model_selection.train_test_split(features, labels, test_size=0.3, random_state=42)

param_grid_kn = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'],
                 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size': [10, 30, 50]}
param_grid_ada = {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [50, 80, 100]}
param_grid_rand = {'n_estimators': [10, 20, 30], 'max_features': ['auto', 'sqrt', 'log2'],
                   'min_samples_split': [2, 5, 10]}

print '--------------------------------------'
print 'CLASSIFICATION'
#clf = GaussianNB()
#clf = tree.DecisionTreeClassifier(min_samples_split=40)

#svr = RandomForestClassifier()
#clf = GridSearchCV(svr, param_grid_rand)

#svr = AdaBoostClassifier()
#clf = GridSearchCV(svr, param_grid_ada)


##Feature scaling
scaler = preprocessing.StandardScaler().fit(features_train)
scaled_features_train = scaler.transform(features_train)
scaled_features_test = scaler.transform(features_test)

svr = KNeighborsClassifier()
clf = GridSearchCV(svr, param_grid_kn)
print 'KNeighborsClassifier'

t1 = time()
print 'Start training...'
clf.fit(scaled_features_train, labels_train)
print 'Training complete!', 'Training time: ', round(time() - t1, 3)

print 'Best parameters: ', clf.best_params_
print 'Best estimator: ', clf.best_estimator_

pred = clf.predict(scaled_features_test)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print 'Accuracy: ', clf.score(scaled_features_test, labels_test)
print 'Average precision: ', metrics.precision_score(labels_test, pred, average='macro')
print 'Average recall: ', metrics.recall_score(labels_test, pred, average='macro')


#final classificatior configuration

clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', leaf_size=10, algorithm='auto')
#clf = AdaBoostClassifier(n_estimators=50, algorithm='SAMME.R')
#clf = RandomForestClassifier(max_features='auto', min_samples_split=5, n_estimators=20)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
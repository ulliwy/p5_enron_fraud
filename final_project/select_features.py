import numpy as np
import sys
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


def create_barchart(features_list, features_scores, desc, ylabel):
    features_scores = zip(features_list, features_scores)
    features_scores = sorted(features_scores, key=lambda x: x[1], reverse=True)

    features_sorted = [x[0] for x in features_scores]
    fl = np.array(features_sorted)
    sc = np.array([x[1] for x in features_scores])
    xs = np.arange(len(fl))

    plt.figure()
    plt.bar(xs, sc, alpha=0.5)
    plt.xticks(xs + 0.35, fl, rotation='vertical')
    plt.tight_layout()
    plt.ylabel(ylabel)
    plt.title('Feature Scores')
    plt.savefig(desc, bbox_inches='tight')
    return features_sorted


def calc_recall_pres(feature_list, dataset, desc):
    current_feature_list = ['poi']
    precision = []
    recall = []

    features_num = []
    all_precisions = []
    all_recalls = []

    for i in range(0, len(feature_list)):
        current_feature_list.append(feature_list[i])
        data = featureFormat(dataset, current_feature_list, sort_keys=True)

        labels, features = targetFeatureSplit(data)
        iterator = StratifiedShuffleSplit(10, random_state=42)

        for train_idx, test_idx in iterator.split(features, labels):
            features_train = []
            features_test = []
            labels_train = []
            labels_test = []
            for j in train_idx:
                features_train.append(features[j])
                labels_train.append(labels[j])
            for k in test_idx:
                features_test.append(features[k])
                labels_test.append(labels[k])

            scaler = preprocessing.StandardScaler().fit(features_train)
            scaled_features_train = scaler.transform(features_train)
            scaled_features_test = scaler.transform(features_test)

            clf = KNeighborsClassifier()
            clf.fit(scaled_features_train, labels_train)

            pred = clf.predict(scaled_features_test)
            precision.append(metrics.precision_score(labels_test, pred))
            recall.append(metrics.recall_score(labels_test, pred))

        all_precisions.append(sum(precision) / float(len(precision)))
        all_recalls.append(sum(recall) / float(len(recall)))
        features_num.append(len(current_feature_list))

    plt.figure()
    plt.scatter(np.array(features_num), np.array(all_precisions))
    plt.plot(np.array(features_num), np.array(all_precisions), label='Precision')
    plt.scatter(features_num, all_recalls, color='green')
    plt.plot(features_num, all_recalls, label='Recall')
    plt.legend(loc='upper right', shadow=True)
    plt.xlabel('Number of Features')
    plt.ylabel('Value')
    plt.title('Precision and Recall vs. Number of Features')

    plt.savefig(desc)


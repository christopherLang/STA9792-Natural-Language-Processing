import csv
import os
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

os.chdir("C:/Users/Christopher Lang/Dropbox/Education/Baruch College/Fall 2017/STA 9792 - Spectial Topics in Statistics/Homework and Project/Final project")

sys.path.append("lib")

from features import SimpleTextFeatures

with open('data/original/fake_or_real_news.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    txt = [i for i in reader]

corpus = list()
labels = list()
for i in txt:
    # corpus.append("\n".join([i['title']]))
    corpus.append("\n".join([i['title'], i['text']]))
    labels.append(i['label'])
#
feature_obj = SimpleTextFeatures(corpus, labels, split=[0.60, 0.30, 0.10])

features_bin = feature_obj.get_feature()
features_freq = feature_obj.get_feature(binary=False)

# kNN Model fitting
k_param = list(range(1, 51))
# metrics = ['minkowski']
metrics = ['minkowski', 'euclidean', 'manhattan']

knn_test_result = list()

for k in tqdm(k_param):
    for a_metric in metrics:
        row_result = dict()
        model = KNeighborsClassifier(n_neighbors=k, metric=a_metric)

        if a_metric not in ['euclidean', 'manhattan']:
            model.fit(features_bin['features'][0], features_bin['labels'][0])
            predict_label = model.predict(features_bin['features'][0])
            e = confusion_matrix(features_bin['labels'][0], predict_label)
        else:
            model.fit(features_freq['features'][0], features_freq['labels'][0])
            predict_label = model.predict(features_freq['features'][0])
            e = confusion_matrix(features_freq['labels'][0], predict_label)

        n_train_correct = sum(e.diagonal())
        n_train_total = sum(sum(e))

        if a_metric not in ['euclidean', 'manhattan']:
            predict_label = model.predict(features_bin['features'][1])
            e = confusion_matrix(features_bin['labels'][1], predict_label)

        else:
            predict_label = model.predict(features_freq['features'][1])
            e = confusion_matrix(features_freq['labels'][1], predict_label)

        n_test_correct = sum(e.diagonal())
        n_test_total = sum(sum(e))

        row_result = dict()
        row_result['k'] = k
        row_result['metric'] = a_metric
        row_result['training_acc'] = n_train_correct / n_train_total
        row_result['testing_acc'] = n_test_correct / n_test_total

        knn_test_result.append(row_result)

with open("diagnostics/knn_classifier-test_result.csv", 'w') as f:
    writer = csv.DictWriter(f, ['k', 'metric', 'training_acc', 'testing_acc'],
                            lineterminator="\n")

    writer.writeheader()
    writer.writerows(knn_test_result)

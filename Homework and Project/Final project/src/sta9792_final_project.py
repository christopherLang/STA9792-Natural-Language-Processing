import csv
import os
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from nltk.corpus import stopwords

os.chdir("C:/Users/Christopher Lang/Dropbox/Education/Baruch College/Fall 2017/STA 9792 - Spectial Topics in Statistics/Homework and Project/Final project")

sys.path.append("lib")

# This lib file is appended with submission
from textutils import data_splitter

# Load csv text file that contains articles
with open('data/original/fake_or_real_news.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    txt = [i for i in reader]

# Create a corpus, in this case using only titles
corpus = list()
labels = list()
for i in txt:
    corpus.append("\n".join([i['title']]))
    # corpus.append("\n".join([i['title'], i['text']]))
    labels.append(i['label'])

# # kNN Model fitting ---------------------------------------------------------
# Range of parameters to test "best" knn model
k_param = list(range(1, 101))
ngram_combo = [(1, 1), (2, 2), (3, 3), (1, 3), (2, 3)]
stop_words = stopwords.words('english')
metrics = ['minkowski', 'euclidean']

knn_test_result = list()

for k in tqdm(k_param):
    for ngram_param in ngram_combo:
        for a_metric in metrics:
            features_tfidf_words = TfidfVectorizer('content',
                                                   stop_words=stop_words,
                                                   ngram_range=ngram_param)

            tfidf_feat = features_tfidf_words.fit_transform(corpus)
            data_split_indices = data_splitter(tfidf_feat.shape[0],
                                               [0.60, 0.30, 0.10])

            training_data = tfidf_feat[data_split_indices[0], ]
            testing_data = tfidf_feat[data_split_indices[1], ]
            validation_data = tfidf_feat[data_split_indices[2], ]

            training_label = [labels[i] for i in data_split_indices[0]]
            testing_label = [labels[i] for i in data_split_indices[1]]
            validation_label = [labels[i] for i in data_split_indices[2]]

            row_result = dict()
            model = KNeighborsClassifier(n_neighbors=k, metric=a_metric)

            model.fit(training_data, training_label)
            predict_label = model.predict(training_data)
            e = confusion_matrix(training_label, predict_label)

            n_train_correct = sum(e.diagonal())
            n_train_total = sum(sum(e))

            predict_label = model.predict(testing_data)
            e = confusion_matrix(testing_label, predict_label)

            n_test_correct = sum(e.diagonal())
            n_test_total = sum(sum(e))

            row_result = dict()
            row_result['k'] = k
            row_result['metric'] = a_metric
            row_result['ngram_param'] = "{0}--{1}".format(ngram_param[0],
                                                          ngram_param[1])
            row_result['training_acc'] = n_train_correct / n_train_total
            row_result['testing_acc'] = n_test_correct / n_test_total

            knn_test_result.append(row_result)

with open("diagnostics/knn_classifier-tfidf-test_result.csv", 'w') as f:
    writer = csv.DictWriter(f, ['k', 'metric', 'ngram_param', 'training_acc',
                                'testing_acc'],
                            lineterminator="\n")

    writer.writeheader()
    writer.writerows(knn_test_result)

# get final model and confuision matrix ---------------------------------------
features_tfidf_words = TfidfVectorizer('content',
                                       stop_words=stop_words,
                                       ngram_range=(1, 1))

tfidf_feat = features_tfidf_words.fit_transform(corpus)
data_split_indices = data_splitter(tfidf_feat.shape[0],
                                   [0.60, 0.30, 0.10])

training_data = tfidf_feat[data_split_indices[0], ]
testing_data = tfidf_feat[data_split_indices[1], ]
validation_data = tfidf_feat[data_split_indices[2], ]

training_label = [labels[i] for i in data_split_indices[0]]
testing_label = [labels[i] for i in data_split_indices[1]]
validation_label = [labels[i] for i in data_split_indices[2]]

model = KNeighborsClassifier(n_neighbors=26, metric='euclidean')

model.fit(training_data, training_label)
predict_label = model.predict(validation_data)
e = confusion_matrix(validation_label, predict_label)

n_valid_correct = sum(e.diagonal())
n_valid_total = sum(sum(e))

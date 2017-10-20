import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import string
import random
from tqdm import tqdm

class SimpleTextFeatures(object):
    def __init__(self, corpus, labels=None, split=None, cleaner=None,
                 featext=None, rmStopwords=True):
        self.corpus = corpus
        self.labels = labels
        self.split = split

        if cleaner is None:
            self.cleaners = list()
            self.cleaners.append({'level': 'article',
                                  'name': 'Sentencer tokenizer',
                                  'fun': self.__sentence_tokenizer})
            self.cleaners.append({'level': 'sentence',
                                  'name': 'Word tokenizer',
                                  'fun': self.__word_tokenizer})
            self.cleaners.append({'level': 'token',
                                  'name': 'Lower casing tokens',
                                  'fun': self.__lowercase})
            self.cleaners.append({'level': 'token',
                                  'name': 'Special token remover',
                                  'fun': self.__string_remover})
            self.cleaners.append({'level': 'token',
                                  'name': 'Special character remover',
                                  'fun': self.__char_remover})
            self.cleaners.append({'level': 'token',
                                  'name': 'Word stemming',
                                  'fun': self.__stemmer})
            self.cleaners.append({'level': 'token',
                                  'name': 'Number replacer',
                                  'fun': self.__number_replacer})

            if rmStopwords is True:
                self.cleaners.append({'level': 'token',
                                      'name': 'Stopword remover',
                                      'fun': self.__stopword_remover})
        else:
            if isinstance(cleaner, list) is not True:
                raise TypeError("cleaner should be a list of functions")

            else:
                self.cleaners = cleaner

        self.cleaner_cached = False

        if featext is None:
            self.featext = list()
            self.featext.append({'level': 'sentence',
                                 'name': 'Gen unigram feature',
                                 'fun': self.__unigram_feature})
            self.featext.append({'level': 'sentence',
                                 'name': 'Gen bigram feature',
                                 'fun': self.__bigram_feature})
        else:
            self.featext = featext

        self.string_to_rm = set()
        self.string_to_rm.update(set(string.punctuation))
        self.string_to_rm.add("''")
        self.string_to_rm.add("n't")
        self.string_to_rm.add("'s")
        self.string_to_rm.add("'re")
        self.string_to_rm.add(".")

        self.stopwords = set(stopwords.words('english'))

        self.char_to_rm = set()
        self.char_to_rm.add('”')
        self.char_to_rm.add('“')
        self.char_to_rm.add('–')
        self.char_to_rm.add('`')

    def set_cleaners(self, cleaners):
        self.cleaners = cleaners

    def append_cleaner(self, cleaner, level):
        self.cleaners.append(cleaner)

    def insert_cleaner(self, cleaner, cleaner_index, level):
        self.cleaners.insert(cleaner_index, cleaner)

    def extend_cleaner(self, cleaners):
        self.cleaners.extend(cleaners)

    def get_cleaners(self):
        return self.cleaners

    def __sentence_tokenizer(self, txt):
        sentences = nltk.sent_tokenize(txt)

        return sentences

    def __word_tokenizer(self, sentence):
        tokens = [nltk.word_tokenize(i) for i in sentence]

        return tokens

    def __lowercase(self, token):
        if token.isupper():
            # Check if ALL characters are uppercase
            # If so, assume acronym, and keep as is
            return token
        else:
            return token.lower()

    def __stemmer(self, token):
        stemmer = nltk.stem.PorterStemmer()

        return stemmer.stem(token)

    def __string_remover(self, token):
        if token in self.string_to_rm:
            r = ''
        else:
            r = token

        return r

    def __char_remover(self, token):
        for i in self.char_to_rm:
            token = token.replace(i, '')

        return token

    def __unigram_feature(self, sent_tok):
        feat = dict()

        for feature in sent_tok:
            if feature in feat.keys():
                feat[feature] += 1
            else:
                feat[feature] = 1

        return feat

    def __bigram_feature(self, sent_tok):
        feat = dict()
        bigrams = nltk.bigrams(sent_tok)
        bigrams = ["_".join(i) for i in bigrams]

        for feature in bigrams:
            if feature in feat.keys():
                feat[feature] += 1
            else:
                feat[feature] = 1

        return feat

    def __stopword_remover(self, token):
        if token in self.stopwords:
            r = ''
        else:
            r = token

        return r

    def __number_replacer(self, token):
        r = None

        try:
            r = float(token)
            r = "|####|"

        except ValueError:
            r = token

        return r

    def data_splitter(self, length, prop, seed=12345):
        random.seed(seed)

        if isinstance(length, int) is not True:
            raise TypeError("parameter length should be an integer")

        sizes = [round(i * length) for i in prop]

        if sum(sizes) != length:
            while True:
                if sum(sizes) == length:
                    break
                else:
                    index = random.sample(range(len(sizes)), 1)[0]
                    sizes[index] -= 1

        document_indices = list(range(length))
        random.shuffle(document_indices)

        result = list()
        initial_i = 0
        for data_size in sizes:
            r = [document_indices[i] for i in range(initial_i,
                                                    initial_i + data_size)]
            result.append(r)

            initial_i = data_size

        return result

    def feature_collapse(self, list_of_features):
        features = list()
        for a_doc in list_of_features:
            doc_features = dict()

            for a_feature_set in a_doc:
                for a_feature in a_feature_set:
                    if a_feature in doc_features.keys():
                        doc_features[a_feature] += 1
                    else:
                        doc_features[a_feature] = 1

            features.append(doc_features)

        return features

    def __get_labels(self, index_splits):
        if self.split is not None:
            result = list()

            for a_index_split in index_splits:
                labels = [self.labels[i] for i in a_index_split]
                result.append(np.array(labels, np.str))
        else:
            result = self.labels

        return result

    def get_feature(self, verbose=True, binary=True):
        if self.featext is None or len(self.featext) == 0:
            raise ValueError("feature functions is None or empty")

        cleaned_text = self.corpus

        if self.cleaners is None or len(self.cleaners) == 0:
            pass

        else:
            for a_cleaner_fun in self.cleaners:
                level = a_cleaner_fun['level']
                cleaner_fun = a_cleaner_fun['fun']

                if level == 'article':
                    if verbose is True:
                        cleaned_text = [cleaner_fun(i)
                                        for i in tqdm(cleaned_text)]
                    else:
                        cleaned_text = [cleaner_fun(i) for i in cleaned_text]

                if level == 'sentence':
                    if verbose is True:
                        cleaned_text = [cleaner_fun(i)
                                        for i in tqdm(cleaned_text)]
                    else:
                        cleaned_text = [cleaner_fun(i) for i in cleaned_text]

                if level == 'token':
                    if verbose is True:
                        cleaned_text = [[[cleaner_fun(z) for z in j]
                                         for j in i]
                                        for i in tqdm(cleaned_text)]
                    else:
                        cleaned_text = [[[cleaner_fun(z) for z in j]
                                         for j in i]
                                        for i in cleaned_text]

        cleaned_text = [[[z for z in j if z != ''] for j in i]
                        for i in cleaned_text]

        features = [list() for _ in range(len(cleaned_text))]
        for i in range(len(cleaned_text)):
            a_doc = cleaned_text[i]

            for a_extractor_fun in self.featext:
                featext_fun = a_extractor_fun['fun']
                # execution_name = a_extractor_fun['name']
                level = a_extractor_fun['level']

                new_feature = [featext_fun(i) for i in a_doc]
                features[i].extend(new_feature)

        features = self.feature_collapse(features)

        features = DictVectorizer(dtype=np.int64).fit_transform(features)

        if binary is True:
            features[features > 0] = 1

        labels = None
        if self.split is not None:
            index_splits = self.data_splitter(features.shape[0], self.split)
            labels = self.__get_labels(index_splits)

            result = list()

            for a_index_split in index_splits:
                result.append(features[a_index_split, ])
        else:
            result = [features]
            labels = [self.labels]

        return {'features': result, 'labels': labels}

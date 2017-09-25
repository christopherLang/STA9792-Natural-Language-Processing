import re
import os
import sys
from collections import defaultdict

workdir = "C:/Users/Christopher Lang/Dropbox/Education/Baruch College/Fall 2017/STA 9792 - Spectial Topics in Statistics/Homework and Project/Homework 2"


def main():
    os.chdir(workdir)
    with open("nytimes-iphone.txt", "r") as f:
        article1 = f.readlines()

    with open("bbc-americanism.txt", "r") as f:
        article2 = f.readlines()

    tokenized1 = clean_and_tokenize(article1)

    unigram1 = [ngram(i, 1) for i in tokenized1]
    unigram1 = flatten(unigram1)
    unigram1_freq = frequency(unigram1)

    bigram1 = [ngram(i, 2) for i in tokenized1]
    bigram1 = flatten(bigram1)
    bigram1_freq = frequency(bigram1)

    tokenized2 = clean_and_tokenize(article2)

    unigram2 = [ngram(i, 1) for i in tokenized2]
    unigram2 = flatten(unigram2)
    unigram2_freq = frequency(unigram2)

    bigram2 = [ngram(i, 2) for i in tokenized2]
    bigram2 = flatten(bigram2)
    bigram2_freq = frequency(bigram2)


def clean_and_tokenize(list_of_str):
    r = [i for i in list_of_str if i != "\n"]
    r = [re.sub("(\n)+$", "", i) for i in r]
    r = [re.split("\s+", i) for i in r]
    r = [[j.lower() for j in i] for i in r]
    r = [[re.subn("\W", "", j)[0] for j in i] for i in r]

    return r


def ngram(list_of_tokens, n=1):
    r = list()
    ind_ngram = list()
    n_i = 0

    for i in range(len(list_of_tokens)):
        ind_ngram.append(list_of_tokens[i])
        n_i += 1
        if n_i == n:
            r.append(ind_ngram)
            ind_ngram = list()
            n_i = 0

    r = [tuple(i) for i in r]

    return r


def flatten(list_of_collections):
    r = [gram for line_of_str in list_of_collections for gram in line_of_str]

    return r


def frequency(ngram_list):
    freq_count = defaultdict(int)

    for a_gram in ngram_list:
        freq_count[a_gram] += 1

    freq_count = [(i, j) for i, j in freq_count.items()]
    freq_count.sort(key=lambda x: x[1], reverse=True)

    return freq_count




if __name__ == "__main__":
    main()
    sys.exit()

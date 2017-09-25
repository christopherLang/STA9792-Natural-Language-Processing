# Homework 3 STA9792
# Baruch College, Fall 2017
# Christopher Lang (EMPL ID: 12091305)
# Python 3.5.3 Anaconda Distribution, Windows 10 64-bit
from nltk.corpus import treebank
import nltk
import numpy as np
import sys
import os


def main():
    try:
        scriptdir = os.path.dirname(os.path.realpath(__file__))
        os.chdir(scriptdir)

    except NameError:
        directory = list()
        directory.append('C:/Users/Christopher Lang/Dropbox/Education')
        directory.append('Baruch College/Fall 2017')
        directory.append('STA 9792 - Spectial Topics in Statistics')
        directory.append('Homework and Project/Homework 3')
        directory = "/".join(directory)

        os.chdir(directory)

    from HMMTagger import HMMPOSTagger

    while True:
        try:
            tagged_sents = treebank.tagged_sents().iterate_from(0)
            break
        except LookupError:
            nltk.download('brown')

    tagged_sents = list(tagged_sents)

    pos_tagger = HMMPOSTagger()
    pos_tagger.trains(tagged_sents)
    pos_tagger.finalize()

    np.random.seed(1000)
    subset_sents = np.random.choice(tagged_sents, 100)

    # accuracy is about 69.63%, with 2466 sentences in testing. In sample
    e = pos_tagger.confusion(subset_sents)

    pos_tags = e['pos_tags']
    pos_tags_header = '\t'.join(pos_tags)
    np.savetxt('confusion_matrix_output.tsv', e['confusion_matrix'], fmt='%u',
               delimiter='\t', header=pos_tags_header)


if __name__ == "__main__":
    main()
    sys.exit()

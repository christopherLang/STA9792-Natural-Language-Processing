# Homework 3 STA9792
# Baruch College, Fall 2017
# Christopher Lang (EMPL ID: 12091305)
# Python 3.5.3 Anaconda Distribution, Windows 10 64-bit
import numpy as np
from scipy.sparse import lil_matrix
from tqdm import tqdm
from itertools import groupby as gy
from operator import itemgetter


class HMMPOSTagger(object):
    def __init__(self):
        self.n_obs_states = 0
        self.n_hid_states = 0
        self.word_transition = dict()
        self.pos_transition = dict()
        self.states = set()
        self.pos = set()
        self.word_pos = set()
        self.word_conditionals = dict()
        self.finalized = False
        self.lowercase = True

        self.txtutils = TextUtilities()

    def trains(self, tagged_sentences, total_sent=None):
        for a_tagged_sentence in tqdm(tagged_sentences, total=total_sent):
            self.train(a_tagged_sentence)

    def train(self, tagged_sentence):
        for i in range(len(tagged_sentence)):
            if i == 0:
                word_i = "<<start-sos>>"
                pos_tag_i = "<<start-sos>>"
                word_f = tagged_sentence[i][0]
                pos_tag_f = tagged_sentence[i][1]

            elif i == (len(tagged_sentence) - 1):
                word_i = tagged_sentence[i][0]
                pos_tag_i = tagged_sentence[i][1]
                word_f = "<<end-eof>>"
                pos_tag_f = "<<end-eof>>"

            else:
                word_i = tagged_sentence[i][0]
                pos_tag_i = tagged_sentence[i][1]
                word_f = tagged_sentence[i + 1][0]
                pos_tag_f = tagged_sentence[i + 1][1]

            word_i = self.txtutils.lowercase(word_i)
            word_f = self.txtutils.lowercase(word_f)

            # count of word_f is conditional on word_i ------------------------
            # Once finalized, it is P(word_f | word_i)
            self.__update_matrix(word_i, word_f, self.word_transition)

            # count of pos_tag_f is conditional on pos_tag_i ------------------
            # Once finalized, it is P(pos_tag_f | pos_tag_i)
            self.__update_matrix(pos_tag_i, pos_tag_f, self.pos_transition)

            # count of word_i is conditional on pos_tag_i ---------------------
            # Once finalized, it is P(word_i | pos_tag_i)
            self.__update_matrix(pos_tag_i, word_i, self.word_conditionals)

            # count of word_f is conditional on pos_tag_f ---------------------
            # Once finalized, it is P(word_f | pos_tag_f)
            self.__update_matrix(pos_tag_f, word_f, self.word_conditionals)

    def __update_matrix(self, from_node, to_node, tran_mat):
        entry_template = {'FROM_NODE': [{'word': 'TO_NODE', 'count': 1}]}
        entry_template['FROM_NODE'][0]['word'] = to_node

        if from_node not in tran_mat.keys():
            entry_template[from_node] = entry_template.pop('FROM_NODE')

            tran_mat.update(entry_template)

        else:
            current_entry = tran_mat[from_node]

            to_node_exists = [True if to_node == i['word'] else False
                              for i in current_entry]

            if any(to_node_exists):
                true_index = [i for i, x in enumerate(to_node_exists) if x][0]

                current_entry[true_index]['count'] += 1

            else:
                current_entry.append(entry_template['FROM_NODE'][0])

    def finalize(self):
        # self.pos_transition will be a matrix (dict with name lookup)
        # where each row is POS tags found, and each column is POS tags found
        # Row is conditional POS tags; given POS found in row, the probability
        # of finding POS in column is contained in the matrix entries

        # Row POS tags transitions to column POS tags, or that the column POS
        # tag follows the row POS tag found in the training corpus
        self.pos_transition = self.__finalize_mat(self.pos_transition)

        # self.word_conditionals will be a matrix (dict with name lookup)
        # where each row is a POS tag found, and each column is a word found
        # Row is conditional POS tags; given POS found in row, the probability
        # of finding a word in column is contained in the matrix entries

        # Row POS tags transitions to column words. It sums and normalizes
        # tags and their found words P(word | tag)
        self.word_conditionals = self.__finalize_mat(self.word_conditionals)

        # self.word_transition will be a matrix (dict with name lookup)
        # where each row is a word found, and each column is a word found
        # Row is conditional word found; given a word found in row, the
        # probability of finding a word in column is contained in the matrix
        # entries

        # Row words transitions to column words, or that the column words
        # follows the row words found in the training corpus
        self.word_transition = self.__finalize_mat(self.word_transition)

        self.finalized = True
        self.n_hid_states = len(self.pos_transition['token_indices']['row'])
        self.n_obs_states = len(self.word_transition['token_indices']['row'])

    def __finalize_mat(self, adjacency):
        token_indices = self.__create_token_index(adjacency)
        edgelist = self.__create_edgelist(adjacency)

        matrix_shape = (len(token_indices['row']),
                        len(token_indices['col']))

        transition_matrix = lil_matrix(matrix_shape, dtype=np.float64)

        for node1, node2, prob in edgelist:
            node1_index = token_indices['row'][node1]
            node2_index = token_indices['col'][node2]
            transition_matrix[node1_index, node2_index] = prob

        # transition_matrix = transition_matrix.tocsc(copy=True)
        return {'matrix': transition_matrix, 'token_indices': token_indices}

    def __create_token_index(self, adjacency):
        token_row_index = dict()
        token_column_index = dict()

        row_keys = set()
        column_keys = set()
        for from_key, to_values in adjacency.items():
            row_keys.add(from_key)
            column_keys.update(set([i['word'] for i in to_values]))

        row_keys.add("<<end-eof>>")
        column_keys.add("<<end-eof>>")
        row_keys.add("<<start-sos>>")
        column_keys.add("<<start-sos>>")

        row_keys = list(row_keys)
        column_keys = list(column_keys)
        row_keys.sort()
        column_keys.sort()

        for i, key in enumerate(row_keys):
            token_row_index[key] = i

        for i, key in enumerate(column_keys):
            token_column_index[key] = i

        return {'row': token_row_index, 'col': token_column_index}

    def __create_edgelist(self, adjacency):
        edgelist = list()

        for key in adjacency:
            transition_to = adjacency[key]
            r = [[key, i['word'], i['count']] for i in transition_to]
            total_count = sum([i[2] for i in r])

            for i in r:
                i[2] /= total_count

            edgelist.extend(r)

        edgelist.sort(key=lambda x: x[0])

        return(edgelist)

    def tag_sentence(self, sentence):
        if self.lowercase is True:
            sentence = [self.txtutils.lowercase(i) for i in sentence]

        words_conditional = self.word_conditionals

        sentence_idx = [words_conditional['token_indices']['col'][i]
                        for i in sentence]

        words_conditional = words_conditional['matrix'][:, sentence_idx]

        viterbi = lil_matrix((self.n_hid_states, len(sentence)),
                             dtype=np.float64)
        backtrace = lil_matrix((self.n_hid_states, len(sentence)),
                               dtype=np.int32)

        # Initialize with start tag
        start = self.pos_transition['token_indices']['row']["<<start-sos>>"]
        for i in range(self.n_hid_states):
            pos_tran_prob = self.pos_transition['matrix'][start, i]
            viterbi[i, 0] = pos_tran_prob * words_conditional[i, 0]

            backtrace[i, 0] = 0

        # Begin iteration over sentence tokens
        for t_i in range(1, len(sentence)):
            for p_i in range(self.n_hid_states):
                new_vprob = list()
                new_btrac = list()
                for p_i2 in range(self.n_hid_states):
                    q_v = (viterbi[p_i2, t_i - 1] *
                           self.pos_transition['matrix'][p_i2, p_i] *
                           words_conditional[p_i, t_i])
                    b_v = (viterbi[p_i2, t_i - 1] *
                           self.pos_transition['matrix'][p_i2, p_i])

                    new_vprob.append(q_v)
                    new_btrac.append(b_v)

                viterbi[p_i, t_i] = max(new_vprob)
                backtrace[p_i, t_i] = new_btrac.index(max(new_btrac))

        qf = self.pos_transition['token_indices']['col']['<<end-eof>>']

        big_T = len(sentence) - 1

        viterbi[qf, big_T] = max(viterbi[:, big_T].toarray().ravel() *
                                 self.pos_transition['matrix'][:, qf].
                                 toarray().ravel())

        backtrace[qf, big_T] = (viterbi[:, big_T].toarray().ravel() *
                                self.pos_transition['matrix'][:, qf].toarray().
                                ravel()).argmax()

        pos_classes = list()
        for i in range(1, len(sentence)):
            index = backtrace[:, i].toarray().ravel().max()

            pos_indices = self.pos_transition['token_indices']['row'].items()
            for pos_tag, pos_index in pos_indices:
                if pos_index == index:
                    pos_classes.append(pos_tag)
                    break

        result = list()
        for a_tok, a_pos_tag in zip(sentence, pos_classes):
            result.append((a_tok, a_pos_tag))

        return result

    def __assess_performance(self, tagged_sentence):
        original_tokens = [i[0] for i in tagged_sentence]
        original_pos = [i[1] for i in tagged_sentence]

        predicted_pos = self.tag_sentence(original_tokens)
        predicted_pos = [i[1] for i in predicted_pos]

        result = list()
        for a_og_pos, a_pr_pos in zip(original_pos, predicted_pos):
            result.append((a_og_pos, a_pr_pos))

        return result

    def confusion(self, tagged_sentences):
        result = list()
        for a_tagged_sentence in tqdm(tagged_sentences):
            result.extend(self.__assess_performance(a_tagged_sentence))

        gy_itemgetter = itemgetter(0, 1)

        result.sort(key=gy_itemgetter)

        final_result = list()
        for pos_compare, perf in gy(result, gy_itemgetter):
            perf = list(perf)
            final_result.append((pos_compare[0], pos_compare[1], len(perf)))

        all_pos = set()
        for i in final_result:
            all_pos.add(i[0])
            all_pos.add(i[1])

        all_pos = list(all_pos)
        all_pos.sort()

        all_pos = [i for i in enumerate(all_pos)]
        all_pos = {pos: index for index, pos in all_pos}

        confusion_matrix = lil_matrix((len(all_pos), len(all_pos)),
                                      dtype=np.int64)

        for i in final_result:
            row_index = all_pos[i[0]]
            col_index = all_pos[i[1]]

            confusion_matrix[row_index, col_index] = np.int64(i[2])

        confusion_matrix = confusion_matrix.todense()

        result_statistic = dict()
        result_statistic['accuracy'] = confusion_matrix.diagonal().sum()
        result_statistic['accuracy'] /= confusion_matrix.sum()
        result_statistic['nobs'] = confusion_matrix.sum()
        confusion_matrix.diagonal

        return {'confusion_matrix': confusion_matrix, 'stat': result_statistic,
                'pos_tags': all_pos}


class TextUtilities(object):
    def __init__(self):
        pass

    def lowercase(self, token):
        if token.isupper():
            # Check if ALL characters are uppercase
            # If so, assume acronym, and keep as is
            return token
        else:
            return token.lower()

import re
from collections import defaultdict, Counter
from typing import List

import nltk
import numpy as np
from nltk.corpus import stopwords
from squad.build_squad_data import SquadCorpus

from data_processing.paragraph_qa import ParagraphAndQuestion
from data_processing.text_features import is_number
from utils import flatten_iterable


class AnswerCat(object):
    def __init__(self, para: ParagraphAndQuestion, answer_tokens: List[str],
                 any_num: List[bool], careful_num: List[bool],
                 is_cap: List[bool], is_upper: List[bool]):
        self.para = para
        self.any_num = any_num
        self.careful_num = careful_num
        self.is_cap = is_cap
        self.is_upper = is_upper
        self.answer = answer_tokens

    def get_answer_str(self):
        output = list(self.answer)
        for i in range(len(output)):
            if self.careful_num[i]:
                output[i] = "NUM<%s>" % output[i]
            elif self.is_cap[i]:
                output[i] = "Cap<%s>" % output[i]
            elif self.is_upper[i]:
                output[i] = "Up<%s>" % output[i]
        return output


def answer_stats(ans_list, name, fn):
    any = 0
    for a in ans_list:
        if fn(a):
            any += 1
    print("%s %d (%.4f)" % (name, any, any/len(ans_list)))


def show_rare_words():
    print("Loading...")
    word_counts = defaultdict(list)
    corpus = SquadCorpus()
    train = corpus.get_train_docs()

    print("Counting...")
    for doc in train:
        doc_counter = Counter()
        for para in doc.paragraphs:
            for question in para.questions:
                doc_counter.update(question.words)
            for sent in para.context:
                doc_counter.update(sent)
        for k,v in doc_counter.items():
            word_counts[k.lower()].append(v)

    for word,v in word_counts.items():
        v = np.array(v)
        v.sort()
        v = v[::-1]
        word_counts[word] = v

    rare = [k for k,v in word_counts.items() if ((v.sum()-v.max()) < 10 and v.sum() > 10)]

    # np.random.shuffle(rare)
    rare.sort(key=lambda x: -word_counts[x].sum())
    for r in rare[:500]:
        v = word_counts[r]
        print("%s %s (total=%d, n_articles=%d)" % (r, str(v[:4]), v.sum(), len(v)))

    print("%d words, %d (%.4f) are rare" % (len(word_counts), len(rare), len(rare)/len(word_counts)))

    n_total = sum(x.sum() for x in word_counts.values())
    n_rare = sum(word_counts[x].sum() for x in rare)
    print("Of %d tokens, %d (%.4f) are rare" % (
        n_total,n_rare, n_rare/n_total))



def show_heuristic_ne():
    print("Loading data...")
    corpus = SpanCorpus("squad")
    data = corpus.get_train_docs()

    vecs = corpus.get_pruned_word_vecs("glove.6B.100d")

    article_occ = Counter()
    upper_counts = Counter()
    lower_counts = Counter()

    print("Computing counts")
    # for doc in data:
    #     article_words = Counter()
    #     for para in doc.paragraphs:
    #         for sent in para.context:
    #             article_words.update(sent)
    #         for q in para.questions:
    #             article_words.update(q.words)
    #
    #     upper_counts += article_words
    #     for k,v in article_words.items():
    #         lower_counts[k.lower()] += v
    #     article_occ.update(article_words.keys())

    data = split_docs(data)

    np.random.shuffle(data)
    stop = set(stopwords.words('english'))
    stop.update([])

    print("Start")
    sent_counts = []
    question_counts = []
    ans_counts = []
    found = Counter()
    punc_regex = re.compile("^\W+$")

    for i,para in enumerate(data):
        ans = para.answer[0]
        if ans.sent_start != ans.sent_end:
            continue
        sent = para.context[ans.sent_start]
        a_start, a_end = ans.word_start, ans.word_end
        n_found = 0
        a_found = 0
        q_found = 0
        for i, word in enumerate(sent):
            if is_number(word):
            # if word.isupper() and punc_regex.match(word) is None and len(word) > 1:
            # if word[0].isupper() and word[1:].islower() and word.lower() not in stop:
            # if upper_counts[word]/lower_counts[word.lower()] > 0.9:
                n_found += 1
                if a_start <= i <= a_end:
                    a_found += 1
                found[word] += 1

        for word in para.question:
            if is_number(word):
            # if word.isupper() and punc_regex.match(word) is None and len(word) > 1:
            # if word[0].isupper() and word[1:].islower() and word.lower() not in stop:
            #     if upper_counts[word]/lower_counts.get(word.lower(),0) > 0.9:
                q_found += 1
                found[word] += 1
                # print(word)

        question_counts.append(q_found)
        ans_counts.append(a_found)
        sent_counts.append(n_found)

    for counts in [sent_counts, ans_counts, question_counts]:
        counts = np.array(counts)
        print("of %d points %d (%.4f) has any, %d total, mean %.4f per occurence" % (len(data), np.sum(counts > 0),
                                                                                     np.sum(counts > 0)/len(data),
                                                                                     np.sum(counts),
                                                                                     np.mean(counts[counts > 0])))
    n_in_vec = sum(v for k,v in found.items() if k.lower() in vecs)
    total = sum(found.values())
    print(n_in_vec, total, n_in_vec/total)

    # keys = list(found.keys())
    # np.random.shuffle(keys)
    #
    # for k in keys[:500]:
    #     print(k)


def show_answer_ne():
    print("Loading data...")
    corpus = SpanCorpus("squad")
    data = corpus.get_dev()
    np.random.shuffle(data)
    data = data[:500]
    print("Processing...")
    counts = []

    for i,para in enumerate(data):
        # answer_tokens = flatten_lists(para.context)[para.answer[0].para_word_start:para.answer[0].para_word_end+1]
        ans = para.answer[0]
        if ans.sent_start != ans.sent_end:
            continue
        sent = para.context[ans.sent_start]
        # print(" ".join(sent))
        chunked_sentences = nltk.ne_chunk(nltk.pos_tag(nltk.tokenize.wordpunct_tokenize(" ".join(sent))), binary=True)

        ents = extract_entity_names(chunked_sentences)
        counts.append(len(ents))

    counts = np.array(counts)

    print("of %d points %d (%.4f) has any, %d total, mean %.4f per occurence" % (len(data), np.sum(counts > 0),
                                                                                 np.sum(counts > 0)/len(data),
                                                                                 np.sum(counts),
                                                                                 np.mean(counts[counts > 0])))

def extract_entity_names(t):
    ents = []
    for subtree in t.subtrees(filter=lambda x: x.label() == 'NE'):
        ents.append(subtree.leaves())
    return ents


def catagorize_answers():
    print("Loading data...")
    corpus = SpanCorpus("squad")
    data = corpus.get_train()

    any_num_regex = re.compile("^.*[\d].*$")
    answers = []

    print("Processing...")

    for para in data:
        answer_tokens = flatten_iterable(para.context)[para.answer[0].para_word_start:para.answer[0].para_word_end + 1]
        any_numeric = [any_num_regex.match(t) is not None for t in answer_tokens]
        careful_num = [is_number(t) is not None for t in answer_tokens]
        cap = [t[0].isupper() and t[1:].islower() is not None for t in answer_tokens]
        upper = [t.isupper() for t in answer_tokens]
        answers.append(AnswerCat(para, answer_tokens, any_numeric, careful_num, cap, upper))

    answer_stats(answers, "cap", lambda x: np.any(x.is_cap))
    answer_stats(answers, "exact", lambda x: np.any(x.any_num))

    for ans in answers:
        # keys = np.logical_and(np.array(ans.is_cap), np.array(ans.any_num))
        keys = ans.is_cap
        if np.any(keys):
            print(" ".join(ans.para.question))
            print(" ".join(ans.get_answer_str()))




if __name__ == "__main__":
    # catagorize_answers()
    # show_heuristic_ne()
    show_rare_words()
    # main(SpanCorpus("squad"), "glove.6B.100d")
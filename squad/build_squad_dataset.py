import argparse
import json
import urllib
from os import listdir
from os.path import expanduser, join, exists
from typing import List

import numpy as np

import config
from squad.squad_data import Question, Document, Paragraph
from data_processing.span_data import ParagraphSpan, ParagraphSpans
from data_processing.text_utils import get_paragraph_tokenizer, convert_to_spans, post_split_tokens, clean_text, \
    get_word_span, space_re
from utils import flatten_iterable

"""
Script to build a corpus from SQUAD training data with `DocumentSpans` answers 
"""


def clean_title(title):
    """ Squad titles use URL escape formatting, this method undos it to get the wiki-title"""
    return urllib.parse.unquote(title).replace("_", " ")


def parse_squad_data(source, name, tokenizer="NLTK") -> List[Document]:
    sent_tokenize, word_tokenize = get_paragraph_tokenizer(tokenizer)

    with open(source, 'r') as f:
        source_data = json.load(f)

    for article_ix, article in enumerate(source_data['data']):
        article_ix = "%s-%d" % (name, article_ix)

        paragraphs = []

        for para_ix, para in enumerate(article['paragraphs']):
            questions = []
            context = para['context']

            # list of sentences, each of which is a list of words
            text = list(map(word_tokenize, sent_tokenize(context)))
            for i, sent in enumerate(text):
                text[i] = post_split_tokens(sent)

            # recover (start, end) for each token relative to the raw `context`
            text_spans = convert_to_spans(context, text)

            # Clean text
            for i, sent in enumerate(text):
                text[i] = [clean_text(x) for x in sent]

            n_words = sum(len(sentence) for sentence in text)

            flat_text = [word for sent in text for word in sent]
            sent_lens = [len(sent) for sent in text]

            for question_ix, question in enumerate(para['qas']):
                question_text = word_tokenize(question['question'])
                question_text = [clean_text(x) for x in question_text]
                answer_text = []
                answer_spans = []
                for answer_ix, answer in enumerate(question['answers']):
                    answer_raw = answer['text']

                    answer_start = answer['answer_start']
                    answer_stop = answer_start + len(answer_raw)

                    word_ixs = get_word_span(text_spans, answer_start, answer_stop)

                    sent_start, word_start = word_ixs[0]
                    sent_end, word_end = word_ixs[-1]
                    first_word = text[sent_start][word_start]
                    first_word_span = text_spans[sent_start][word_start]
                    last_word = text[sent_end][word_end]
                    last_word_span = text_spans[sent_end][word_end]

                    char_start = answer_start - first_word_span[0]
                    char_end = answer_stop - last_word_span[0]

                    # Sanity check to ensure we can rebuild the answer using the word and char indices
                    # Since we might not be able to "undo" the tokenizing exactly we might not be able to exactly
                    # rebuild 'answer_raw', so just we check that we can rebuild the answer minus spaces
                    if len(word_ixs) == 1:
                        if first_word[char_start:char_end] != answer_raw:
                            raise ValueError()
                    else:
                        rebuild = first_word[char_start:]
                        for word_ix in word_ixs[1:-1]:
                            rebuild += text[word_ix[0]][word_ix[1]]
                        rebuild += last_word[:char_end]
                        if rebuild != space_re.sub("", clean_text(answer_raw)):
                            raise ValueError(rebuild + " " + answer_raw)

                    para_word_start = word_start + sum(sent_lens[:sent_start])
                    para_word_end = word_end + sum(sent_lens[:sent_end])
                    if text[sent_start][word_start] != flat_text[para_word_start]:
                        raise RuntimeError()
                    if text[sent_end][word_end] != flat_text[para_word_end]:
                        raise RuntimeError()

                    span = ParagraphSpan(
                        sent_start, word_start, char_start,
                        sent_end, word_end, char_end,
                        word_start + sum(sent_lens[:sent_start]),
                        word_end + sum(sent_lens[:sent_end]),
                        answer_raw)
                    if span.para_word_end >= n_words or \
                                    span.para_word_start >= n_words:
                        raise RuntimeError()
                    answer_spans.append(span)

                questions.append(Question(question_text, ParagraphSpans(answer_spans), question['id']))

            span_ar = np.array(list(((int(x[0]), int(x[1])) for x in flatten_iterable(text_spans))), dtype=np.int32)

            paragraphs.append(Paragraph(text, questions, article_ix, para_ix, context, span_ar))

        yield Document(article_ix, article["title"], None, paragraphs)


def main():
    parser = argparse.ArgumentParser()
    source_dir = join(expanduser("~"), "data", "squad")
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--name", default="squad")
    parser.add_argument("--tokenizer", default="NLTK", type=str)

    args = parser.parse_args()
    target_dir = join(config.CORPUS_DIR, args.name)

    if exists(target_dir) and len(listdir(target_dir)) > 0:
        raise ValueError("Files already exist in " + target_dir + ", if you really want to override delete them"
                                                                       "and then and re-run")
    print("Parsing train...")
    train = list(parse_squad_data(join(source_dir, "train-v1.1.json"), "train", args.tokenizer))

    print("Parsing dev...")
    dev = list(parse_squad_data(join(source_dir, "dev-v1.1.json"), "dev", args.tokenizer))

    print("Saving...")
    SquadCorpus.make_corpus(target_dir, train, dev, None)
    print("Done")


if __name__ == "__main__":
    main()
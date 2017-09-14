import argparse
import json
import urllib
from os import listdir
from os.path import expanduser, join, exists
from typing import List

import numpy as np

import config
from squad.squad_data import Question, Document, Paragraph, SquadCorpus
from data_processing.span_data import ParagraphSpan, ParagraphSpans
from data_processing.text_utils import get_word_span, space_re, NltkAndPunctTokenizer
from utils import flatten_iterable

"""
Script to build a corpus from SQUAD training data 
"""


def clean_title(title):
    """ Squad titles use URL escape formatting, this method undos it to get the wiki-title"""
    return urllib.parse.unquote(title).replace("_", " ")


def parse_squad_data(source, name, tokenizer, pbar=True) -> List[Document]:
    with open(source, 'r') as f:
        source_data = json.load(f)

    it = source_data['data']
    if pbar:
        # Optional in case the client wants to run w/o installing tqdm (i.e. for codalab scripts)
        from tqdm import tqdm
        it = tqdm(it)
    for article_ix, article in enumerate(it):
        article_ix = "%s-%d" % (name, article_ix)

        paragraphs = []

        for para_ix, para in enumerate(article['paragraphs']):
            questions = []
            context = para['context']

            # list of sentences, each of which is a list of words
            tokenized = tokenizer.tokenize_with_inverse(context)
            text, text_spans = tokenized.text, tokenized.spans
            flat_text = flatten_iterable(text)

            n_words = sum(len(sentence) for sentence in text)

            sent_lens = [len(sent) for sent in text]

            for question_ix, question in enumerate(para['qas']):
                question_text = tokenizer.tokenize_sentence(question['question'])

                answer_spans = []
                for answer_ix, answer in enumerate(question['answers']):
                    answer_raw = answer['text']

                    answer_start = answer['answer_start']
                    answer_stop = answer_start + len(answer_raw)

                    word_ixs = get_word_span(text_spans, answer_start, answer_stop)

                    first_word = flat_text[word_ixs[0]]
                    first_word_span = text_spans[word_ixs[0]]
                    last_word = flat_text[word_ixs[-1]]
                    last_word_span = text_spans[word_ixs[-1]]

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
                            rebuild += flat_text[word_ix]
                        rebuild += last_word[:char_end]
                        if rebuild != space_re.sub("", tokenizer.clean_text(answer_raw)):
                            raise ValueError(rebuild + " " + answer_raw)

                    # Find the sentence with in-sentence offset
                    sent_start, sent_end, word_start, word_end = None, None, None, None
                    on_word = 0
                    for sent_ix, sent in enumerate(text):
                        next_word = on_word + len(sent)
                        if on_word <= word_ixs[0] < next_word:
                            sent_start = sent_ix
                            word_start = word_ixs[0] - on_word
                        if on_word <= word_ixs[-1] < next_word:
                            sent_end = sent_ix
                            word_end = word_ixs[-1] - on_word
                            break
                        on_word = next_word

                    if text[sent_start][word_start] != flat_text[word_ixs[0]]:
                        raise RuntimeError()
                    if text[sent_end][word_end] != flat_text[word_ixs[-1]]:
                        raise RuntimeError()

                    span = ParagraphSpan(
                        sent_start, word_start, char_start,
                        sent_end, word_end, char_end,
                        word_ixs[0], word_ixs[-1],
                        answer_raw)
                    if span.para_word_end >= n_words or \
                                    span.para_word_start >= n_words:
                        raise RuntimeError()
                    answer_spans.append(span)

                questions.append(Question(question['id'], question_text, ParagraphSpans(answer_spans)))

            paragraphs.append(Paragraph(text, questions, article_ix, para_ix, context, text_spans))

        yield Document(article_ix, article["title"], paragraphs)


def main():
    parser = argparse.ArgumentParser()
    source_dir = join(expanduser("~"), "data", "squad")
    parser.add_argument('-s', "--source_dir", default=source_dir)

    args = parser.parse_args()
    source_dir = args.source_dir
    target_dir = join(config.CORPUS_DIR, SquadCorpus.NAME)
    tokenzier = NltkAndPunctTokenizer()

    if exists(target_dir) and len(listdir(target_dir)) > 0:
        raise ValueError("Files already exist in " + target_dir)

    print("Parsing train...")
    train = list(parse_squad_data(join(source_dir, "train-v1.1.json"), "train", tokenzier))

    print("Parsing dev...")
    dev = list(parse_squad_data(join(source_dir, "dev-v1.1.json"), "dev", tokenzier))

    print("Saving...")
    SquadCorpus.make_corpus(train, dev)
    print("Done")


if __name__ == "__main__":
    main()
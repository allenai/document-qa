import argparse
import json
import urllib
from os import listdir, mkdir
from os.path import expanduser, join, exists
from typing import List

from tqdm import tqdm

from docqa import config
from docqa.squad.squad_data import Question, Document, Paragraph, SquadCorpus
from docqa.data_processing.span_data import ParagraphSpan, ParagraphSpans
from docqa.data_processing.text_utils import get_word_span, space_re, NltkAndPunctTokenizer
from docqa.utils import flatten_iterable

"""
Script to build a corpus from SQUAD training data 
"""


def clean_title(title):
    """ Squad titles use URL escape formatting, this method undoes it to get the wiki-title"""
    return urllib.parse.unquote(title).replace("_", " ")


def parse_squad_data(source, name, tokenizer, use_tqdm=True) -> List[Document]:
    with open(source, 'r') as f:
        source_data = json.load(f)

    if use_tqdm:
        iter_files = tqdm(source_data['data'], ncols=80)
    else:
        iter_files = source_data['data']

    for article_ix, article in enumerate(iter_files):
        article_ix = "%s-%d" % (name, article_ix)

        paragraphs = []

        for para_ix, para in enumerate(article['paragraphs']):
            questions = []
            context = para['context']

            tokenized = tokenizer.tokenize_with_inverse(context)
            # list of sentences + mapping from words -> original text index
            text, text_spans = tokenized.text, tokenized.spans
            flat_text = flatten_iterable(text)

            n_words = sum(len(sentence) for sentence in text)

            for question_ix, question in enumerate(para['qas']):
                # There are actually some multi-sentence questions, so we should have used
                # tokenizer.tokenize_paragraph_flat here which would have produced slighy better
                # results in a few cases. However all the results we report were
                # done using `tokenize_sentence` so I am just going to leave this way
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

                    # Sanity check these as well
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
    parser = argparse.ArgumentParser("Preprocess SQuAD data")
    basedir = join(expanduser("~"), "data", "squad")
    parser.add_argument("--train_file", default=join(basedir, "train-v1.1.json"))
    parser.add_argument("--dev_file", default=join(basedir, "dev-v1.1.json"))

    if not exists(config.CORPUS_DIR):
        mkdir(config.CORPUS_DIR)

    target_dir = join(config.CORPUS_DIR, SquadCorpus.NAME)
    if exists(target_dir) and len(listdir(target_dir)) > 0:
        raise ValueError("Files already exist in " + target_dir)

    args = parser.parse_args()
    tokenzier = NltkAndPunctTokenizer()

    print("Parsing train...")
    train = list(parse_squad_data(args.train_file, "train", tokenzier))

    print("Parsing dev...")
    dev = list(parse_squad_data(args.dev_file, "dev", tokenzier))

    print("Saving...")
    SquadCorpus.make_corpus(train, dev)
    print("Done")


if __name__ == "__main__":
    main()
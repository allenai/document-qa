import argparse
import pickle
from typing import List, Optional

import numpy as np
import tensorflow as tf

from data_processing.batching import get_clustered_batches
from data_processing.paragraph_qa import get_doc_input_spec, compute_document_voc, DocParagraphAndQuestion, Document
from data_processing.span_data import get_best_in_sentence_span, compute_span_f1
from data_processing.wiki_online import SquadWikiArticles
from trainer import ModelDir
from squad.aligned_wiki_qa import WikiArticleQaCorpus
from squad.squad import SquadCorpus
from squad.squad_official_evaluation import f1_score as squad_official_f1_score
from utils import CachingResourceLoader, ResourceLoader, flatten_iterable

"""
Should be considered deprecated until we make a better version similar the triviaq sript
"""

class QuestionResult(object):
    def __init__(self, question_id, right_para, doc_span_f1, doc_text_f1,
                 para_span_f1, para_text_f1):
        self.question_id = question_id
        self.right_para = right_para
        self.para_span_f1 = para_span_f1
        self.para_text_f1 = para_text_f1
        self.doc_span_f1 = doc_span_f1
        self.doc_text_f1 = doc_text_f1


class QuestionAnswer(object):
    def __init__(self, question_id, paragraph_num, doc_span, para_span):
        self.question_id = question_id
        self.paragraph_num = paragraph_num
        self.doc_span = tuple(doc_span)
        self.para_span = tuple(para_span)

    def __repr__(self):
        return "Answer-" + self.question_id


class QuestionAnswerFull(object):
    def __init__(self, question_id, span_vals, spans):
        self.question_id = question_id
        self.span_vals = span_vals
        self.spans = spans


class JointPredictor(object):

    def __init__(self, model_dir, test_data: List[Document], batch_size, loader: ResourceLoader, sess, ema=False):
        self.batch_size = batch_size
        model = model_dir.get_model()
        checkpoint = model_dir.get_latest_checkpoint()

        spec = get_doc_input_spec(None, [test_data])
        voc = compute_document_voc(test_data)

        model.set_input_spec(spec, voc, loader)


        model_pred = model.get_prediction()

        tf.train.Saver().restore(sess, checkpoint)

        if ema:
            print("Restoring EMA variables")
            ema = tf.train.ExponentialMovingAverage(0)
            saver = tf.train.Saver({ema.average_name(x): x for x in tf.trainable_variables()})
            saver.restore(sess, checkpoint)

        self.model = model
        self.unnormlzied_start_logit = tf.exp(model_pred.prediction.start_logits)
        self.unnormlzied_end_logit = tf.exp(model_pred.prediction.end_logits)

        sess.graph.finalize()

    def eval(self, doc: Document, sample, sess):
        doc_questions = flatten_iterable([x.questions for x in doc.paragraphs])
        if sample is not None:
            np.random.shuffle(doc_questions)
            doc_questions = doc_questions[:sample]

        joint_data = []
        # Store the best span, and best span  value, per question, per paragraph
        # there can be a lot of these so we store compactly in numpy array
        question_spans = {}
        question_span_values = {}

        n_paras = len(doc.paragraphs)
        for q in doc_questions:
            question_spans[q.question_id] = np.full((n_paras, 2), -1, dtype=np.int32)
            question_span_values[q.question_id] = np.full(n_paras, -1, dtype=np.float32)
            for p in doc.paragraphs:
                joint_data.append(DocParagraphAndQuestion(q.words, None, q.question_id, p))

        print("On: %s, with %d pairs" % (doc.title, len(joint_data)))

        cluster = lambda x: sum(len(s) for s in x.context)
        for _, _, batch in get_clustered_batches(joint_data, self.batch_size, cluster, 1, 0, False, None, True):
            enc = self.model.encode(batch, False)
            l1, l2 = sess.run([self.unnormlzied_start_logit, self.unnormlzied_end_logit], feed_dict=enc)

            for i, point in enumerate(batch):
                para_num = point.paragraph_num
                question_id = point.question_id
                pred, val = get_best_in_sentence_span(l1[i], l2[i], [len(s) for s in point.context])
                question_spans[question_id][para_num] = pred
                question_span_values[question_id][para_num] = val

        for q in doc_questions:
            yield q, question_spans[q.question_id], question_span_values[q.question_id]


def eval(model_dir: ModelDir, batch_size: int, test_data: List[Document], loader: ResourceLoader,
         n_docs: Optional[int], sample_per_doc: Optional[int], output: Optional[str]=None, ema=False):

    if n_docs is not None:
        np.random.shuffle(test_data)
        test_data = test_data[:n_docs]

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    print("Setting up model")
    pred = JointPredictor(model_dir, test_data, batch_size, loader, sess, ema)

    question_results = []
    question_answers = []
    print("Start eval")

    for doc in test_data:
        quid_to_paragraph = {}
        quid_to_question = {}
        para_id_to_paragraph = {}

        for para in doc.paragraphs:
            para_id_to_paragraph[para.paragraph_num] = para
            for q in para.questions:
                quid_to_question[q.question_id] = q
                quid_to_paragraph[q.question_id] = para

        for q, spans, span_vals in pred.eval(doc, sample_per_doc, sess):
            correct_para_num = quid_to_paragraph[q.question_id].paragraph_num

            predicted_paragraph = np.argmax(span_vals)
            answer = QuestionAnswer(q.question_id, predicted_paragraph,
                                    spans[predicted_paragraph], spans[correct_para_num])

            question_answers.append(QuestionAnswerFull(q.question_id, span_vals, spans))

            q_span_f1 = 0
            q_text_f1 = 0
            para_text = doc.paragraphs[correct_para_num].get_original_text(*answer.para_span)

            for ans in q.answer:
                q_span_f1 = max(q_span_f1, compute_span_f1((ans.para_word_start, ans.para_word_end), answer.para_span))
                q_text_f1 = max(q_text_f1, squad_official_f1_score(para_text, ans.text))

            doc_span_f1 = 0
            doc_text_f1 = 0
            if answer.paragraph_num == correct_para_num:
                doc_span_f1, doc_text_f1 = q_span_f1, q_text_f1
            else:
                doc_text = doc.paragraphs[answer.paragraph_num].get_original_text(*answer.doc_span)
                for ans in q.answer:
                    doc_text_f1 = max(doc_text_f1, squad_official_f1_score(doc_text, ans.text))

            question_results.append(QuestionResult(q.question_id, predicted_paragraph == correct_para_num,
                                                   doc_span_f1, doc_text_f1, q_span_f1, q_text_f1))

        print("CorrectPara=%.4f, DocSpanF1=%.4f, DocTextF1=%.4f, ParaSpanF1=%.4f, ParaTextF1=%.4f" % (
            np.mean([x.right_para for x in question_results]),
            np.mean([x.doc_span_f1 for x in question_results]),
            np.mean([x.doc_text_f1 for x in question_results]),
            np.mean([x.para_span_f1 for x in question_results]),
            np.mean([x.para_text_f1 for x in question_results])
        ))

    if output is not None:
        with open(output, "wb") as f:
            pickle.dump(question_answers, f)
    print('Done')


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model', help='name of output to exmaine')
    parser.add_argument('-d', '--n_doc', type=int, default=None)
    parser.add_argument('-q', '--n_questions_per_doc', type=int, default=None)
    parser.add_argument('-b', '--batch_size', type=int, default=200)
    parser.add_argument('-c', '--corpus', choices=["squad", "wiki-aligned-squad", "wiki-sub-squad"], default="squad-dev")
    parser.add_argument('-o', '--output', type=str, default=None)
    args = parser.parse_args()

    model_dir = ModelDir(args.model)
    corpus = SquadCorpus()

    if args.corpus == "squad":
        data = corpus.get_dev_docs()
    elif args.corpus == "wiki-aligned-squad":
        articles = SquadWikiArticles()
        data = WikiArticleQaCorpus(corpus, articles, False, 0.15).get_dev_docs()
    elif args.corpus == "wiki-sub-squad":
        articles = SquadWikiArticles()
        data = WikiArticleQaCorpus(corpus, articles, True, 0.15).get_dev_docs()
    else:
        raise RuntimeError()

    model_dir.get_latest_checkpoint()
    loader = CachingResourceLoader(corpus.get_pruned_word_vecs)
    eval(model_dir, args.get_fixed_batch_size, data, loader, args.n_doc, args.n_questions_per_doc, args.output, True)


def tmp():
    with open("/tmp/test.pkl", "rb") as f:
        data = pickle.load(f)
        print(data)

if __name__ == "__main__":
    main()
    # tmp()





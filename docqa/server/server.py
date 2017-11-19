import argparse
import logging
import time
import ujson
from os import environ
from typing import List

import numpy as np
import tensorflow as tf
from sanic import Sanic, response
from sanic.exceptions import ServerError
from sanic.response import json

from docqa.data_processing.document_splitter import MergeParagraphs, ShallowOpenWebRanker
from docqa.data_processing.span_data import top_disjoint_spans
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
from docqa.model import Model, Prediction
from docqa.model_dir import ModelDir
from docqa.nn.span_prediction import BoundaryPrediction
from docqa.server.qa_system import WebParagraph, QaSystem
from docqa.text_preprocessor import WithIndicators
from docqa.utils import ResourceLoader, LoadFromPath


"""
Server for the demo. The server uses the async/await framework so the API calls and 
downloads can be done asynchronously, allowing the server can answer other queries in the
meantime. 
"""


log = logging.getLogger('server')


class AnswerSpan(object):
    def __init__(self, conf: float, start: int, end: int):
        self.conf = conf
        self.start = start
        self.end = end

    def to_json(self):
        # make sure we are using python types
        return dict(conf=float(self.conf), start=int(self.start), end=int(self.end))


class WebParagraphWithSelectedAnswers(object):
    def __init__(self, url: str, source_name: str, original_text: str, answers: List[AnswerSpan]):
        self.source_url = url
        self.source_name = source_name
        self.original_text = original_text
        self.answers = answers

    def to_json(self):
        return dict(source_url=self.source_url, source_name=self.source_name,
                    text=self.original_text, answers=[x.to_json() for x in self.answers])


def select_answers(paras: List[WebParagraph], span_scores, bound, n_spans) -> List[WebParagraphWithSelectedAnswers]:
    """
    Selects the top `n_spans` non-overlapping spans of at most length `bound` from each paragraph,
    returns the resulting paragraphs sorted by most confidence answer
    """
    out = []
    for para, score in zip(paras, span_scores):
        n_tokens = len(para.spans)
        # Score can contain other stuff due to padding
        score = score[:n_tokens, :n_tokens]

        top_n, top_n_scores = top_disjoint_spans(score, bound, n_spans, para.spans)
        answers = []
        for score, (s, e) in zip(top_n_scores, top_n):
            s = para.spans[s][0]
            e = para.spans[e][1]
            answers.append(AnswerSpan(score, s, e))
        out.append(WebParagraphWithSelectedAnswers(para.source_url, para.source_name, para.original_text,
                                                   answers))
    out.sort(key=lambda x:-x.answers[0].conf)
    return out

ipso = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore 
magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea 
commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat 
nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""


class DummyQa():
    async def answer_question(self, question: str):
        return self.get_random_answer()

    def get_random_answer(self):
        time.sleep(1)
        para = NltkAndPunctTokenizer().tokenize_with_inverse(ipso)
        para1 = WebParagraph(para.text, ipso, para.spans, 0, 0, 0, "source1", "fake_url1")
        para2 = WebParagraph(para.text, ipso, np.array(para.spans), 0, 0, 0, "source2", "fake_url2")
        span_scores = np.random.normal(size=(2, len(para.spans), len(para.spans))) * 5
        return span_scores, [para1, para2]

    def answer_with_doc(self, question: str, doc: str):
        return self.get_random_answer()


class RandomPredictor(Model):
    def __init__(self, std, pre=None):
        print("Using a RandomPredictor, I hope you are debugging")
        self.std = std
        self.preprocessor = pre
        self.context_size = None

    def init(self, train_data, resource_loader: ResourceLoader):
        pass

    def set_input_spec(self, input_spec, voc, word_vec_loader):
        self.context_size = tf.placeholder(tf.int32, (2,))

    def set_inputs(self, datasets, resource_loader: ResourceLoader):
        self.context_size = tf.placeholder(tf.int32, (2,))

    def get_placeholders(self):
        return [self.context_size]

    def encode(self, examples, is_train: bool):
        return {self.context_size: np.array([len(examples),
                                             max(x.n_context_words for x in examples)])}

    def get_predictions_for(self, input_tensors) -> Prediction:
        sz = input_tensors[self.context_size]
        start_logits = tf.random_normal(sz, stddev=self.std)
        end_logits = tf.random_normal(sz, stddev=self.std)
        return BoundaryPrediction(None, None, start_logits, end_logits, None)


def main():
    parser = argparse.ArgumentParser(description='Run the demo server')
    parser.add_argument('model', help='Models to use')

    parser.add_argument('-v', '--voc', help='vocab to use, only words from this file will be used')
    parser.add_argument('-t', '--tokens', type=int, default=400,
                        help='Number of tokens to use per paragraph')
    parser.add_argument('--vec_dir', help='Location to find word vectors')
    parser.add_argument('--n_paragraphs', type=int, default=12,
                        help="Number of paragraphs to run the model on")
    parser.add_argument('--span_bound', type=int, default=8,
                        help="Max span size to return as an answer")

    parser.add_argument('--tagme_api_key', help="Key to use for TAGME (tagme.d4science.org/tagme)")
    parser.add_argument('--bing_api_key', help="Key to use for bing searches")
    parser.add_argument('--tagme_thresh', default=0.2, type=float)
    parser.add_argument('--no_wiki', action="store_true", help="Dont use TAGME")
    parser.add_argument('--n_web', type=int, default=10, help='Number of web docs to fetch')
    parser.add_argument('--blacklist_trivia_sites', action="store_true",
                        help="Don't use trivia websites")
    parser.add_argument('-c', '--wiki_cache', help="Cache wiki articles in this directory")

    parser.add_argument('--n_dl_threads', type=int, default=5,
                        help="Number of threads to download documents with")
    parser.add_argument('--request_timeout', type=int, default=60)
    parser.add_argument('--download_timeout', type=int, default=25)
    parser.add_argument('--workers', type=int, default=1,
                        help="Number of server workers")
    parser.add_argument('--debug', default=None, choices=["random_model", "dummy_qa"])

    args = parser.parse_args()
    span_bound = args.span_bound

    if args.tagme_api_key is not None:
        tagme_api_key = args.tagme_api_key
    else:
        tagme_api_key = environ.get("TAGME_API_KEY")

    if args.bing_api_key is not None:
        bing_api_key = args.bing_api_key
    else:
        bing_api_key = environ.get("BING_API_KEY")
        if bing_api_key is None and args.n_web > 0:
            raise ValueError("If n_web > 0 you must give a BING_API_KEY")

    if args.debug is None:
        model = ModelDir(args.model)
    else:
        model = RandomPredictor(5, WithIndicators())

    if args.vec_dir is not None:
        loader = LoadFromPath(args.vec_dir)
    else:
        loader = ResourceLoader()

    if args.debug == "dummy_qa":
        qa = DummyQa()
    else:
        qa = QaSystem(
            args.wiki_cache,
            MergeParagraphs(args.tokens),
            ShallowOpenWebRanker(args.n_paragraphs),
            args.voc,
            model,
            loader,
            bing_api_key,
            tagme_api_key=tagme_api_key,
            n_dl_threads=args.n_dl_threads,
            blacklist_trivia_sites=args.blacklist_trivia_sites,
            download_timeout=args.download_timeout,
            span_bound=span_bound,
            tagme_threshold=None if args.no_wiki else args.tagme_thresh,
            n_web_docs=args.n_web
        )

    logging.propagate = False
    formatter = logging.Formatter("%(asctime)s: %(levelname)s: %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.DEBUG)

    app = Sanic()
    app.config.REQUEST_TIMEOUT = args.request_timeout

    @app.route("/answer")
    async def answer(request):
        try:
            question = request.args["question"][0]
            if question == "":
                return response.json(
                    {'message': 'No question given'},
                    status=400
                )
            spans, paras = await qa.answer_question(question)
            answers = select_answers(paras, spans, span_bound, 10)
            return json([x.to_json() for x in answers])
        except Exception as e:
            log.info("Error: " + str(e))

            raise ServerError("Server Error", status_code=500)

    @app.route('/answer-from', methods=['POST'])
    async def answer_from(request):
        try:
            args = ujson.loads(request.body.decode("utf-8"))
            question = args.get("question")
            if question is None or question == "":
                return response.json(
                    {'message': 'No question given'},
                    status=400)
            doc = args["document"]
            if len(doc) > 500000:
                raise ServerError("Document too large", status_code=400)
            spans, paras = qa.answer_with_doc(question, doc)
            answers = select_answers(paras, spans, span_bound, 10)
            return json([x.to_json() for x in answers])
        except Exception as e:
            log.info("Error: " + str(e))
            raise ServerError("Server Error", status_code=500)

    app.static('/', './docqa//server/static/index.html')
    app.static('/about.html', './docqa//service/static/about.html')
    app.run(host="0.0.0.0", port=8000, workers=args.workers, debug=False)


if __name__ == "__main__":
    main()

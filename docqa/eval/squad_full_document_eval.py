import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from docqa import trainer
from docqa.data_processing.qa_training_data import ContextAndQuestion, Answer, ParagraphAndQuestionDataset
from docqa.data_processing.span_data import TokenSpans
from docqa.data_processing.text_utils import NltkPlusStopWords, ParagraphWithInverse
from docqa.dataset import FixedOrderBatcher
from docqa.eval.triviaqa_full_document_eval import RecordParagraphSpanPrediction
from docqa.model_dir import ModelDir
from docqa.squad.document_rd_corpus import get_doc_rd_doc
from docqa.squad.squad_data import SquadCorpus
from docqa.squad.squad_document_qa import SquadTfIdfRanker
from docqa.utils import ResourceLoader, flatten_iterable


class RankedParagraphQuestion(ContextAndQuestion):

    def __init__(self, question: List[str], answer: Optional[Answer],
                 question_id: str, paragraph: ParagraphWithInverse,
                 rank: int, paragraph_number: int):
        super().__init__(question, answer, question_id)
        self.paragraph = paragraph
        self.rank = rank
        self.paragraph_number = paragraph_number

    def get_original_text(self, para_start, para_end):
        return self.paragraph.get_original_text(para_start, para_end)

    def get_context(self):
        return flatten_iterable(self.paragraph.text)

    @property
    def n_context_words(self) -> int:
        return sum(len(s) for s in self.paragraph.text)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on document-level SQuAD')
    parser.add_argument('model', help='model to use')
    parser.add_argument('output', type=str,
                        help="Store the per-paragraph results in csv format in this file")
    parser.add_argument('-n', '--n_sample', type=int, default=None,
                        help="(for testing) run on a subset of questions")
    parser.add_argument('-s', '--async', type=int, default=10,
                        help="Encoding batch asynchronously, queueing up to this many")
    parser.add_argument('-a', '--answer_bound', type=int, default=17,
                        help="Max answer span length")
    parser.add_argument('-p', '--n_paragraphs', type=int, default=None,
                        help="Max number of paragraphs to use")
    parser.add_argument('-b', '--batch_size', type=int, default=200,
                        help="Batch size, larger sizes can be faster but uses more memory")
    parser.add_argument('-c', '--corpus', choices=["dev", "train", "doc-rd-dev"], default="dev")
    parser.add_argument('--ema', action="store_true",
                        help="Use EMA weights")
    args = parser.parse_args()

    model_dir = ModelDir(args.model)
    print("Loading data")

    questions = []
    ranker = SquadTfIdfRanker(NltkPlusStopWords(True),
                              args.n_paragraphs, force_answer=False)

    if args.corpus == "doc-rd-dev":
        docs = SquadCorpus().get_dev()
        if args.n_sample is not None:
            docs.sort(key=lambda x:x.doc_id)
            np.random.RandomState(0).shuffle(docs)
            docs = docs[:args.n_sample]

        print("Fetching document reader docs...")
        doc_rd_versions = get_doc_rd_doc(docs)
        print("Ranking and matching with questions...")
        for doc in tqdm(docs):
            doc_questions = flatten_iterable(x.questions for x in doc.paragraphs)
            paragraphs = doc_rd_versions[doc.title]
            ranks = ranker.rank([x.words for x in doc_questions], [x.text for x in paragraphs])
            for i, question in enumerate(doc_questions):
                para_ranks = np.argsort(ranks[i])
                for para_rank, para_num in enumerate(para_ranks[:args.n_paragraphs]):
                    # Just use dummy answers spans for these pairs
                    questions.append(RankedParagraphQuestion(question.words,
                                            TokenSpans(question.answer.answer_text, np.zeros((0, 2), dtype=np.int32)),
                                            question.question_id, paragraphs[para_num], para_rank, para_num))
        rl = ResourceLoader()
    else:
        if args.corpus == "dev":
            docs = SquadCorpus().get_dev()
        else:
            docs = SquadCorpus().get_train()
        rl = SquadCorpus().get_resource_loader()

        if args.n_sample is not None:
            docs.sort(key=lambda x:x.doc_id)
            np.random.RandomState(0).shuffle(docs)
            docs = docs[:args.n_sample]

        for q in ranker.ranked_questions(docs):
            for i, p in enumerate(q.paragraphs):
                questions.append(RankedParagraphQuestion(q.question,
                                                         TokenSpans(q.answer_text, p.answer_spans),
                                                         q.question_id,
                                                         ParagraphWithInverse([p.text], p.original_text, p.spans),
                                                         i, p.paragraph_num))

    print("Split %d docs into %d paragraphs" % (len(docs), len(questions)))

    questions = sorted(questions, key=lambda x: (x.n_context_words, len(x.question)), reverse=True)
    for q in questions:
        if len(q.answer.answer_spans.shape) != 2:
            raise ValueError()

    checkpoint = model_dir.get_latest_checkpoint()
    data = ParagraphAndQuestionDataset(questions, FixedOrderBatcher(args.batch_size, True))

    model = model_dir.get_model()
    evaluation = trainer.test(model, [RecordParagraphSpanPrediction(args.answer_bound)],
                              {args.corpus: data}, rl, checkpoint,
                              args.ema, args.async)[args.corpus]

    print("Saving result")
    output_file = args.output

    df = pd.DataFrame(evaluation.per_sample)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()





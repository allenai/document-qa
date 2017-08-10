import pickle
import ujson as json
from collections import Counter
from os import listdir
from os.path import join
from time import perf_counter

import numpy as np
from tqdm import tqdm

from data_processing.document_splitter import MergeParagraphs, TopTfIdf, Truncate
from data_processing.preprocessed_corpus import PreprocessedData
from data_processing.qa_data import Batcher
from data_processing.text_utils import get_paragraph_tokenizer, NltkPlusStopWords, WordNormalizer
from data_processing.word_vectors import load_word_vectors
from eval.triviaqa_eval import TriviaQaOracle

from paragraph_selection.paragraph_selection_featurizer import ParagraphFeatures, ParagraphOrderFeatures, \
    WordMatchingNeighborFeaturizer
from paragraph_selection.paragraph_selection_model import NParagraphsSortKey
from paragraph_selection.paragraph_selection_with_context import SelectionWithContextDatasetBuilder
from paragraph_selection.word_occurance_model import OccuranceFeaturizer, OccuranceDatasetBuilder
from trivia_qa.build_span_corpus import TriviaQaWebDataset, TriviaQaSpanCorpus
from trivia_qa.evidence_corpus import TriviaQaEvidenceCorpusTxt
from trivia_qa.read_data import iter_question_json
from utils import split, flatten_iterable


def test():
    data = []


    t0 = perf_counter()
    docs = []
    for file in listdir("/Users/chrisc/document-qa/data/triviaqa/np-evidence-v2/web/0"):
        arr = np.load(join("/Users/chrisc/document-qa/data/triviaqa/np-evidence-v2/web/0", file))
        docs.append(arr[arr > 0])
    print(perf_counter() - t0)
    #
    #
    t0 = perf_counter()
    docs = []
    for file in listdir("/Users/chrisc/document-qa/data/triviaqa/np-evidence-v3/web/0"):
        data = np.fromfile(join("/Users/chrisc/document-qa/data/triviaqa/np-evidence-v3/web/0", file), dtype=np.int32)
        docs.append(data[data > 0])
        docs.append(np.fromfile(join("/Users/chrisc/document-qa/data/triviaqa/np-evidence-v3/web/0", file), dtype=np.int32))
    print(perf_counter() - t0)
    #
    # t0 = perf_counter()
    # docs = []
    # f = h5py.File("/tmp/corpus.hdf5")["web"]["0"]
    # for group in f:
    #     docs.append(f[group] )
    # print(perf_counter() - t0)

    # t0 = perf_counter()
    # data = np.load("/tmp/corpus-f.npy")
    # print(perf_counter() - t0)

    # t0 = perf_counter()
    # with open("/tmp/corpus-pickle.pkl", "rb") as f:
    #     data = pickle.load(f)
    # print(perf_counter() - t0)

    # source = "/Users/chrisc/document-qa/data/triviaqa/np-evidence/corpus-c.npz"
    # t0 = perf_counter()
    # f = np.load(source)
    # docs = [f[k] for k in f if k != "vocab"]
    # print(perf_counter() - t0)

    # t0 = perf_counter()
    # docs = []
    # for file in listdir("/Users/chrisc/document-qa/data/triviaqa/evidence/web/0"):
    #     with open(join("/Users/chrisc/document-qa/data/triviaqa/evidence/web/0", file), "r") as f:
    #         docs.append(json.load(f))
    # print(perf_counter() - t0)

    t0 = perf_counter()
    files = listdir("/Users/chrisc/Programming/data/trivia-qa/evidence/web/0")
    docs = []
    look_up = dict(the=0)
    for file in files:
        with open(join("/Users/chrisc/Programming/data/trivia-qa/evidence/web/0", file), "r") as f:
            data = []
            text = f.read().split()
            for word in text:
                if word in look_up:
                    pass
            docs.append(data)
    print(perf_counter() - t0)


def test_np():
    data = np.fromfile("/Users/chrisc/document-qa/data/triviaqa/evidence-np/web/0/0_100129.npy", dtype=np.int32)
    with open("/Users/chrisc/document-qa/data/triviaqa/evidence-np/vocab.txt", "r") as f:
        voc = [f.rstrip() for f in f]
    data = data[data > 0]
    # print(len(voc))
    print([voc[i] for i in data])


def show_names():
    data = SquadCorpus()
    train = data.get_train_docs()
    counts = Counter()
    for doc in train:
        for para in doc.paragraphs:
            for sent in para.context:
                counts.update(sent)

    word_counts_lower = Counter()
    for k,v in counts.items():
        word_counts_lower[k.lower()] += v

    named_count = Counter()
    for word, c in counts.items():
        if word[0].isupper() and word[1:].islower() and word[-1] != ".":
            wl = word.lower()
            lc = word_counts_lower[wl]
            if lc == 0 or (c / lc) > 0.90:
                named_count[word] = c

    for w,c in named_count.most_common():
        print("%s: %d" % (w, c))

def build_pruned_vec(question_files, vec_name, tokenizer_name, n_processes):
    _, word_tokenize = get_paragraph_tokenizer(tokenizer_name)
    voc = set()

    for file in question_files:
        print("Loading questions %s..." % file)
        for q in iter_question_json(file):
            q = json.loads(q)
            voc.update(word_tokenize(q["Question"]))

    data = TriviaQaEvidenceCorpusTxt()
    all_evidence_docs = data.list_documents()
    print("Done, scanning corpus...")
    if n_processes == 1:
        data = TriviaQaEvidenceCorpusTxt()
        for x in tqdm(all_evidence_docs):
            voc.update(data.get_document(x, flat=True))
    else:
        from multiprocessing import Pool
        pool = Pool(n_processes)
        np.random.shuffle(all_evidence_docs)
        chunks = split(all_evidence_docs, n_processes)
        output = pool.starmap(build_voc, [(x, 2000) for x in chunks], chunksize=1)
        voc = output[0]
        for x in output[1:]:
            voc.update(x)

    print("Done, loading word vecs...")
    vecs = load_word_vectors(vec_name, voc)
    with open("/tmp/tmp.pkl", "wb") as f:
        pickle.dump(vecs, f)



def check_flat():
    corpus = TriviaQaEvidenceCorpusTxt()
    for doc in tqdm(corpus.list_documents()):
        flat = corpus.get_document(doc, 800, flat=True)
        other = flatten_iterable(flatten_iterable(corpus.get_document(doc)))[:800]
        if flat != other:
            raise ValueError()


def check_oracle():
    from trivia_qa.trivia_qa_eval import f1_score as trivia_f1_score
    # tok = get_paragraph_tokenizer("NLTK_AND_CLEAN")[1]
    # detector = FastNormalizedAnswerDetector()
    corpus = TriviaQaSpanCorpus("web2")
    data = corpus.get_verified()
    f1 = 0
    total = 0
    pairs = flatten_iterable([[(q, x) for x in q.all_docs] for q in data])
    # pairs = [x for x in pairs if x[0].question_id == "qb_6706" and x[1].doc_id == "Animal Farm"]
    np.random.shuffle(pairs)
    for q, doc in tqdm(pairs[:10000]):
        answer = q.answer
        total += 1
        spans = doc.answer_spans[doc.answer_spans[:, 4] < 800]
        if len(spans) == 0:
            continue
        spans = spans[:, 3:]
        point_f1 = 0
        doc_text = corpus.evidence.get_document(doc.doc_id, 800, True)
        for s, e in spans:
            text = " ".join(doc_text[s:e])
            for alias in answer.all_answers:
                point_f1 = max(point_f1, trivia_f1_score(text, alias))
        if point_f1 < 1:
            raise ValueError()
        f1 += point_f1
    print(f1/total)


def check_v2():
    part = "verified"
    with open("data/triviaqa/web/%s.pkl" % part, "rb") as f:
        v2 = pickle.load(f)
    with open("/Users/chrisc/Desktop/web-original/%s.pkl" % part, "rb") as f:
        v1 = pickle.load(f)
    if len(v1) != len(v2):
        raise ValueError()
    if set(x.question_id for x in v1) != set(x.question_id for x in v2):
        raise ValueError()

    quid_map = {q.question_id:q for q in v1}
    for q_v2 in v2:
        other = quid_map[q_v2.question_id]
        for d1, d2 in zip(q_v2.all_docs, other.all_docs):
            if d1.doc_id != d2.doc_id:
                raise ValueError()
            spans2 = d2.answer_spans[:, 3:]
            spans2[:, 1] -= 1
            spans1 = d1.answer_spans
            print(spans2.shape, spans1.shape)
            if not np.all(spans1 == spans2):
                raise ValueError()


def run_oracle():
    # 0.15049065253205357
    oracle = TriviaQaOracle()
    train_batching = Batcher(45, "bucket_context_words_3", True, False)
    eval_batching = Batcher(45, "context_words", False, False)
    data = InMemoryWebQuestions(TriviaQaWebDataset(), train_batching, eval_batching,
                         eval_train=False, eval_verified=False,
                         # splitter=MergeParagraphs(800),
                                splitter=Truncate(800),
                         # paragraph_filter=TopTfIdf(NltkPlusStopWords(punctuation=True), 1),
                         intern=False, per_document=False,
                         sample_dev=2000, sample=(10, 10))

    dataset = data.get_eval()["dev"]
    print(dataset.n_total)
    print(dataset.percent_filtered())
    # train.batcher.truncate_batches = True
    # print(len(flatten_iterable(x[2] for x in train.get_batches(1))))
    # print(oracle.evaluate(flatten_iterable(x[2] for x in train.get_batches(1)), len(train)).scalars)
    print(oracle.evaluate(dataset.data, dataset.n_total).scalars)


def print_questions(question, answers, context, answer_span):
    print(" ".join(question))
    print(answers)
    context = flatten_iterable(context)
    for s,e in answer_span:
        context[s] = "{{{" + context[s]
        context[e] = context[e] + "}}}"
    print(" ".join(context))


def test_nn():
    stop = NltkPlusStopWords(True)
    norm = WordNormalizer(stemmer="wordnet")
    datasets = TriviaQaWebDataset()
    fe = WordMatchingNeighborFeaturizer(stop, norm, "glove.6B.50d")
    splitter = MergeParagraphs(400)

    data = datasets.get_dev()
    np.random.shuffle(data)

    for q in data[:100]:
        for doc in q.all_docs:
            text = datasets.evidence.get_document(doc.doc_id)
            paragraphs = splitter.split(text, doc.answer_spans)
            fe.get_joint_features(q.question, paragraphs)
            return


def tmp_inter():
    stop = NltkPlusStopWords(True)
    norm = WordNormalizer(stemmer="wordnet")
    fe = OccuranceFeaturizer(MergeParagraphs(400), None,
                             [ParagraphOrderFeatures(), ParagraphFeatures()],
                             stop, norm, True, False)

    train_batching = Batcher(30, NParagraphsSortKey(), True, False)
    eval_batching = Batcher(30, NParagraphsSortKey(), False, False)
    datasets = TriviaQaWebDataset()
    data = PreprocessedData(
        datasets, fe,
        OccuranceDatasetBuilder(train_batching, eval_batching),
        eval_on_verified=False,
        sample=20, sample_dev=1,
    )
    data.preprocess(1)
    for _,_,data in data.get_train().get_batches(1):
        for x in data:
            # print(" ".join(x.question))
            # text = datasets.evidence.get_document(x.doc_id, flat=True)
            print([len(o) for o in x.occurances])
            # for (s, e), occ in zip(x.spans, x.occurances):
            #     paragraph = text[s:e+1]
                # print(" ".join(paragraph))
        break
                # print(s, e, len(occ))
            # print(x)


def check():
    corpus = TriviaQaWebDataset()
    train = corpus.get_train()
    dev = corpus.get_dev()
    train_ids = set(q.question_id for q in train)
    dev_ids = set(q.question_id for q in dev)
    print(len(train_ids.intersection(dev_ids)))

# def align_unfiltered():
if __name__ == "__main__":
    check()
    # test_nn()
    # check_v2()
    # check_flat()
    # run_oracle()
    # test_find()
    # all_qs = [
    #  join(TRIVIA_QA, "qa", "wikipedia-train.json"),
    #  join(TRIVIA_QA, "qa", "wikipedia-dev.json"),
    #  join(TRIVIA_QA, "qa", "verified-wikipedia-dev.json"),
    #  join(TRIVIA_QA, "qa", "wikipedia-test-without-answers.json"),
    #  join(TRIVIA_QA, "qa", "web-train.json"),
    #  join(TRIVIA_QA, "qa", "web-dev.json"),
    #  join(TRIVIA_QA, "qa", "verified-web-dev.json"),
    #  join(TRIVIA_QA, "qa", "web-test-without-answers.json"),
    #  join(TRIVIA_QA_UNFILTERED, "unfiltered-web-train.json"),
    #  join(TRIVIA_QA_UNFILTERED, "unfiltered-web-dev.json"),
    #  join(TRIVIA_QA_UNFILTERED, "unfiltered-web-test-without-answers.json"),
    # ]
    # build_pruned_vec(all_qs, "glove.840B.300d", "NLTK_AND_CLEAN", 2)
    # tmp =                   dict(
    #                   verified=join(TRIVIA_QA, "qa", "verified-web-dev.json"),
    #                   dev=join(TRIVIA_QA, "qa", "web-dev.json"),
    #                   train=join(TRIVIA_QA, "qa", "web-train.json"),
    #               )
    # build_file_map(list(tmp.values()))
    # check_answers(join(TRIVIA_QA, "qa", "wikipedia-train.json"))
    # "St.", "Congress" "Washington" "Xbox" "Sir" "Unicode" "Princess" "Moreover" "Jr." "Mr.", "Dr." "Christmas" ,"Inc"
    # months, days of week
    # show_names()
    # with open("data/triviaqa/web/file_map.json", "r") as f:
    #     data = json.load(f)
    # for k,v in data.items():
    #     data[k] = v[:-5]
    # with open("data/triviaqa/web/file_map.json", "w") as f:
    #     json.dump(data, f)



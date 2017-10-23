import json
import sys
import tensorflow as tf
import nltk

from docqa.data_processing.qa_training_data import ParagraphAndQuestionDataset, ContextLenKey
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
from docqa.data_processing.word_vectors import load_word_vectors, load_word_vector_file
from docqa.dataset import ClusteredBatcher
from docqa.squad.build_squad_dataset import parse_squad_data
from docqa.squad.squad_data import split_docs
from docqa.model_dir import ModelDir
from docqa.utils import ResourceLoader


"""
Used to submit our official SQuAD scores via codalab
"""


def run():
    input_data = sys.argv[1]
    output_path = sys.argv[2]
    model_dir = ModelDir("model")
    nltk.data.path.append("nltk_data")

    print("Loading data")
    docs = parse_squad_data(input_data, "", NltkAndPunctTokenizer(), False)
    pairs = split_docs(docs)
    dataset = ParagraphAndQuestionDataset(pairs, ClusteredBatcher(100, ContextLenKey(), False, True))

    print("Done, init model")
    model = model_dir.get_model()
    # small hack, just load the vector file at its expected location rather then using the config location
    loader = ResourceLoader(lambda a, b: load_word_vector_file("glove.840B.300d.txt", b))
    model.set_inputs([dataset], loader)

    print("Done, building graph")
    sess = tf.Session()
    with sess.as_default():
        pred = model.get_prediction()
    best_span = pred.get_best_span(17)[0]

    print("Done, loading weights")
    checkpoint = model_dir.get_latest_checkpoint()
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)
    ema = tf.train.ExponentialMovingAverage(0)
    saver = tf.train.Saver({ema.average_name(x): x for x in tf.trainable_variables()})
    saver.restore(sess, checkpoint)

    print("Done, starting evaluation")
    out = {}
    for batch in dataset.get_epoch():
        enc = model.encode(batch, False)
        spans = sess.run(best_span, feed_dict=enc)
        for (s, e), point in zip(spans, batch):
            out[point.question_id] = point.get_original_text(s, e)

    sess.close()

    print("Done, saving")
    with open(output_path, "w") as f:
        json.dump(out, f)

    print("Mission accomplished!")


if __name__ == "__main__":
    run()




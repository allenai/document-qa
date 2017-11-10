import argparse
import json
from os.path import join

import nltk
import tensorflow as tf

from docqa.data_processing.qa_training_data import ParagraphAndQuestionDataset, ContextLenKey
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
from docqa.data_processing.word_vectors import load_word_vector_file
from docqa.dataset import ClusteredBatcher
from docqa.model_dir import ModelDir
from docqa.squad.build_squad_dataset import parse_squad_data
from docqa.squad.squad_data import split_docs
from docqa.utils import ResourceLoader

"""
Used to submit our official SQuAD scores via codalab
"""


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_data")
    parser.add_argument("output_data")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=100)
    parser.add_argument("--ema", action="store_true")
    args = parser.parse_args()

    input_data = args.input_data
    output_path = args.output_data
    model_dir = ModelDir("model")
    nltk.data.path.append("nltk_data")

    print("Loading data")
    docs = parse_squad_data(input_data, "", NltkAndPunctTokenizer(), False)
    pairs = split_docs(docs)
    dataset = ParagraphAndQuestionDataset(pairs, ClusteredBatcher(args.batch_size, ContextLenKey(), False, True))

    print("Done, init model")
    model = model_dir.get_model()
    # small hack, just load the vector file at its expected location rather then using the config location
    loader = ResourceLoader(lambda a, b: load_word_vector_file("glove.840B.300d.txt", b))
    lm_model = model.lm_model
    basedir = "lm"
    lm_model.lm_vocab_file = join(basedir, "squad_train_dev_all_unique_tokens.txt")
    lm_model.options_file = join(basedir, "options_squad_lm_2x4096_512_2048cnn_2xhighway_skip.json")
    lm_model.weight_file = join(basedir, "squad_context_concat_lm_2x4096_512_2048cnn_2xhighway_skip.hdf5")
    lm_model.embed_weights_file = None

    model.set_inputs([dataset], loader)

    print("Done, building graph")
    sess = tf.Session()
    with sess.as_default():
        pred = model.get_prediction()
    best_span = pred.get_best_span(17)[0]

    all_vars = tf.global_variables() + tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
    dont_restore_names = {x.name for x in all_vars if x.name.startswith("bilm")}
    print(sorted(dont_restore_names))
    vars = [x for x in all_vars if x.name not in dont_restore_names]

    print("Done, loading weights")
    checkpoint = model_dir.get_best_weights()
    if checkpoint is None:
        print("Loading most recent checkpoint")
        checkpoint = model_dir.get_latest_checkpoint()
    else:
        print("Loading best weights")

    saver = tf.train.Saver(vars)
    saver.restore(sess, checkpoint)

    if args.ema:
        ema = tf.train.ExponentialMovingAverage(0)
        saver = tf.train.Saver({ema.average_name(x): x for x in tf.trainable_variables()})
        saver.restore(sess, checkpoint)

    sess.run(tf.variables_initializer([x for x in all_vars if x.name in dont_restore_names]))

    print("Done, starting evaluation")
    out = {}
    for i, batch in enumerate(dataset.get_epoch()):
        if args.n is not None and i == args.n:
            break
        print("On batch: %d" % (i +1))
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




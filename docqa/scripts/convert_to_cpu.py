import argparse
import pickle
from os import mkdir, listdir
from os.path import exists, isfile, join
from shutil import copyfile

import numpy as np
import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

from docqa.data_processing.qa_training_data import ParagraphAndQuestionSpec, ParagraphAndQuestion
from docqa.model_dir import ModelDir
from docqa.nn.recurrent_layers import BiRecurrentMapper, CompatGruCellSpec
from docqa.utils import ResourceLoader

"""
Script to convert out models to cpu version, Due to what appears to be a bug in tf 
(https://github.com/tensorflow/tensorflow/issues/13254) RNNParamsSavable is not working for me, 
if it was we could probably implement this in the Cudnn layers. Instead we complete the transform
manually, which means this will only work for our models.
"""


def convert(model_dir, output_dir, best_weights=False):
    print("Load model")
    md = ModelDir(model_dir)
    model = md.get_model()
    dim = model.embed_mapper.layers[1].n_units
    global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                  initializer=tf.constant_initializer(0), trainable=False)

    print("Setting up cudnn version")
    # global_step = tf.get_variable('global_step', shape=[], dtype='int32', trainable=False)
    sess = tf.Session()
    with sess.as_default():
        model.set_input_spec(ParagraphAndQuestionSpec(1, None, None, 14),
                             {"the"},
                             ResourceLoader(lambda a,b: {"the": np.zeros(300, np.float32)}))

        print("Buiding graph")
        pred = model.get_prediction()

    test_questions = ParagraphAndQuestion(["Harry", "Potter", "was", "written", "by", "JK"],
                                          ["Who", "wrote", "Harry", "Potter", "?"],
                                          None, "test_questions")

    print("Load vars")
    md.restore_checkpoint(sess)

    feed = model.encode([test_questions], False)
    cuddn_out = sess.run([pred.start_logits, pred.end_logits], feed_dict=feed)

    print("Done, copying files...")
    if not exists(output_dir):
        mkdir(output_dir)
    for file in listdir(model_dir):
        if isfile(file) and file != "model.npy":
            copyfile(join(model_dir, file), join(output_dir, file))

    print("Done, mapping tensors...")
    to_save = []
    to_init = []
    for x in tf.trainable_variables():
        if x.name.endswith("/gru_parameters:0"):
            key = x.name[:-len("/gru_parameters:0")]
            fw_params = x
            if "map_embed" in x.name:
                c = cudnn_rnn_ops.CudnnGRU(1, dim, 400)
            elif "chained-out" in x.name:
                c = cudnn_rnn_ops.CudnnGRU(1, dim, dim * 4)
            else:
                c = cudnn_rnn_ops.CudnnGRU(1, dim, dim * 2)
            params_saveable = cudnn_rnn_ops.RNNParamsSaveable(
                c, c.params_to_canonical,
                c.canonical_to_params, [fw_params],
                key)

            for spec in params_saveable.specs:
                if spec.name.endswith("bias_cudnn 0") or \
                        spec.name.endswith("bias_cudnn 1"):
                    # ??? What do these even do?
                    continue
                name = spec.name.split("/")
                name.remove("cell_0")
                if "forward" in name:
                    ix = name.index("forward")
                    name.insert(ix+2, "fw")
                else:
                    ix = name.index("backward")
                    name.insert(ix + 2, "bw")
                del name[ix]

                ix = name.index("multi_rnn_cell")
                name[ix] = "bidirectional_rnn"
                name = "/".join(name)
                v = tf.Variable(sess.run(spec.tensor), name=name)
                to_init.append(v)
                to_save.append(v)

        else:
            to_save.append(x)

    other = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if x not in tf.trainable_variables()]
    print(other)
    sess.run(tf.initialize_variables(to_init))
    saver = tf.train.Saver(to_save + other)
    save_dir = join(output_dir, "save")
    if not exists(save_dir):
        mkdir(save_dir)

    saver.save(sess, join(save_dir, "checkpoint"), sess.run(global_step))

    sess.close()
    tf.reset_default_graph()

    print("Updating model...")
    model.embed_mapper.layers = [model.embed_mapper.layers[0],
                                 BiRecurrentMapper(CompatGruCellSpec(dim))]
    model.match_encoder.layers = list(model.match_encoder.layers)
    other = model.match_encoder.layers[1].other
    other.layers = list(other.layers)
    other.layers[1] = BiRecurrentMapper(CompatGruCellSpec(dim))

    pred = model.predictor.predictor
    pred.first_layer = BiRecurrentMapper(CompatGruCellSpec(dim))
    pred.second_layer = BiRecurrentMapper(CompatGruCellSpec(dim))

    with open(join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    print("Testing...")
    with open(join(output_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    sess = tf.Session()

    model.set_input_spec(ParagraphAndQuestionSpec(1, None, None, 14),
                         {"the"},
                         ResourceLoader(lambda a, b: {"the": np.zeros(300, np.float32)}))
    pred = model.get_prediction()

    print("Rebuilding")
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    feed = model.encode([test_questions], False)
    cpu_out = sess.run([pred.start_logits, pred.end_logits], feed_dict=feed)

    print("These should be close:")
    print([np.allclose(a, b) for a,b in zip(cpu_out, cuddn_out)])
    print(cpu_out)
    print(cuddn_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_model")
    parser.add_argument("output_dir")
    parser.add_argument("--best_weights", action="store_true")
    args = parser.parse_args()
    convert(args.target_model, args.output_dir, args.best_weights)

if __name__ == "__main__":
    main()


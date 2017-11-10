import argparse

import tensorflow as tf

from docqa.data_processing.qa_training_data import ParagraphAndQuestion, ParagraphAndQuestionSpec
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
from docqa.elmo.lm_qa_models import ElmoQaModel
from docqa.model_dir import ModelDir

"""
Script to run a model on user provided question/context input.
Its main purpose is to be an example of how to use the model on new question/context pairs.
"""


def main():
    parser = argparse.ArgumentParser(description="Run an ELMo model on user input")
    parser.add_argument("model", help="Model directory")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("context", help="Context to answer the question with")
    args = parser.parse_args()

    # Tokenize the input, the models expected data to be tokenized using `NltkAndPunctTokenizer`
    # Note the model expects case-sensitive input
    tokenizer = NltkAndPunctTokenizer()
    question = tokenizer.tokenize_paragraph_flat(args.question)
    context = tokenizer.tokenize_paragraph_flat(args.context)

    print("Loading model")
    model_dir = ModelDir(args.model)
    model = model_dir.get_model()
    if not isinstance(model, ElmoQaModel):
        raise ValueError("This script is build to work for ElmoQaModel models only")

    # Important! This tells the language model not to use the pre-computed word vectors,
    # which are only applicable for the SQuAD dev/train sets.
    # Instead the language model will use its character-level CNN to compute
    # the word vectors dynamically.
    model.lm_model.embed_weights_file = None

    # Tell the model the batch size and vocab to expect, This will load the needed
    # word vectors and fix the batch size when building the graph / encoding the input
    print("Setting up model")
    voc = set(question)
    voc.update(context)
    model.set_input_spec(ParagraphAndQuestionSpec(batch_size=1), voc)

    # Now we build the actual tensorflow graph, `best_span` and `conf` are
    # tensors holding the predicted span (inclusive) and confidence scores for each
    # element in the input batch
    print("Build tf graph")
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    with sess.as_default():
        # 17 means to limit the span to size 17 or less
        best_spans, conf = model.get_prediction().get_best_span(17)

    # Now restore the weights, this is a bit fiddly since we need to avoid restoring the
    # bilm weights, and instead load them from the pre-computed data
    all_vars = tf.global_variables() + tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
    lm_var_names = {x.name for x in all_vars if x.name.startswith("bilm")}
    vars = [x for x in all_vars if x.name not in lm_var_names]
    model_dir.restore_checkpoint(sess, vars)

    # Run the initializer of the lm weights, which will load them from the lm directory
    sess.run(tf.variables_initializer([x for x in all_vars if x.name in lm_var_names]))

    # Now the model is ready to run
    # The model takes input in the form of `ContextAndQuestion` objects, for example:
    data = [ParagraphAndQuestion(context, question, None, "user-question1")]

    print("Starting run")
    # The model is run in two steps, first it "encodes" the paragraph/context pairs
    # into numpy arrays, then to use `sess` to run the actual model get the predictions
    encoded = model.encode(data, is_train=False)  # batch of `ContextAndQuestion` -> feed_dict
    best_spans, conf = sess.run([best_spans, conf], feed_dict=encoded)  # feed_dict -> predictions
    print("Best span: " + str(best_spans[0]))
    print("Answer text: " + " ".join(context[best_spans[0][0]:best_spans[0][1]+1]))
    print("Confidence: " + str(conf[0]))


if __name__ == "__main__":
    main()
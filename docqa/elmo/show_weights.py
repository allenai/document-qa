import argparse
import tensorflow as tf
from docqa.model_dir import ModelDir
import numpy as np


def softmax(x):
    x = np.exp(x)
    return x / x.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    model_dir = ModelDir(args.model)
    checkpoint = model_dir.get_best_weights()
    reader = tf.train.NewCheckpointReader(checkpoint)

    if reader.has_tensor("weight_embed_context_lm/layer_0/w"):
        x = "w"
    else:
        x = "ELMo_W_0"

    for i in reader.get_variable_to_shape_map().items():
        print(i)

    input_w = reader.get_tensor("weight_embed_lm/layer_0/%s/ExponentialMovingAverage" % x)
    output_w = reader.get_tensor("weight_lm/layer_0/%s/ExponentialMovingAverage" % x)

    print("Input")
    print(input_w)
    print("(Softmax): " + str(softmax(input_w)))

    print("Output")
    print(output_w)
    print("(Softmax): " + str(softmax(output_w)))

if __name__ == "__main__":
    main()


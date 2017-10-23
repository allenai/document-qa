import argparse
import tensorflow as tf
from docqa.model_dir import ModelDir
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("model")
    args = parser.parse_args()

    model_dir = ModelDir(args.model)
    checkpoint = model_dir.get_best_weights()
    print(checkpoint)
    if checkpoint is None:
        print("Show latest checkpoint")
        checkpoint = model_dir.get_latest_checkpoint()
    else:
        print("Show best weights")

    reader = tf.train.NewCheckpointReader(checkpoint)
    param_map = reader.get_variable_to_shape_map()
    total = 0
    for k in sorted(param_map):
        v = param_map[k]
        print('%s: %s' % (k, str(v)))
        total += np.prod(v)

    print("%d total" % total)



if __name__ == "__main__":
    main()



import argparse
import tensorflow as tf
from model_dir import ModelDir


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("model")
    args = parser.parse_args()

    model_dir = ModelDir(args.model)
    checkpoint = model_dir.get_latest_checkpoint()
    print(checkpoint)

    reader = tf.train.NewCheckpointReader(checkpoint)
    param_map = reader.get_variable_to_shape_map()
    for k, v in param_map.items():
        print('%s: %s' % (k, str(v)))

if __name__ == "__main__":
    main()



import os
import argparse
import tensorflow as tf

# 外部参数配置
parser = argparse.ArgumentParser()
parser.add_argument('--data_url', type=str, default="cache/data", help='training data')
parser.add_argument('--train_url', type=str, default="cache/output", help='training output')
parser.add_argument('--num_gpus', type=int, default=1, help='training gpu num')
ARGS = parser.parse_args()

if __name__ == '__main__':
    model = tf.keras.Model()
    tf.saved_model.save(model, os.path.join(ARGS.train_url, "model"))

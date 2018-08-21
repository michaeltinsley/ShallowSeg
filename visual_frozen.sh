#!/usr/bin/env bash
source activate tensorflow

mkdir -p ./log/quantised/frozen_tensorboard_log

python tensorflow/tensorflow/python/tools/import_pb_to_tensorboard.py \
        --model_dir=./log/quantised/frozen_graph.pb \
        --log_dir=./log/quantised/frozen_tensorboard_log

tensorboard --logdir=./log/quantised/frozen_tensorboard_log
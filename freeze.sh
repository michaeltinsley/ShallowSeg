#!/usr/bin/env bash
source activate tensorflow

mkdir -p ./log/quantised

freeze_graph --input_graph=./log/original/graph.pbtxt \
             --input_checkpoint=./log/original/model.ckpt-32827 \
             --input_binary=false \
             --output_graph=./log/quantised/frozen_graph.pb \
             --output_node=ShallowSeg/Softmax


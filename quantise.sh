#!/usr/bin/env bash
source activate tensorflow

mkdir -p ./log/quantised/

python tensorflow/tensorflow/tools/quantization/quantize_graph.py \
        --input=./log/quantised/frozen_graph.pb \
        --output_node_names="ShallowSeg/Softmax" \
        --print_nodes=True \
        --output=./log/quantised/16bit_quantised_graph.pb \
        --mode=eightbit \
        --bitdepth=16




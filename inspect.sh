#!/usr/bin/env bash
source activate tensorflow


tensorflow/tensorflow/tools/benchmark/benchmark_model \
        --graph=./log/quantised/frozen_graph.pb \
        --input_layer=


bazel build -c opt tensorflow/tools/benchmark:benchmark_model && \
bazel-bin/tensorflow/tools/benchmark/benchmark_model \
--graph=/tmp/inception_graph.pb --input_layer="Mul:0" \
--input_layer_shape="1,299,299,3" --input_layer_type="float" \
--output_layer="softmax:0" --show_run_order=false --show_time=false \
--show_memory=false --show_summary=true --show_flops=true --logtostderr
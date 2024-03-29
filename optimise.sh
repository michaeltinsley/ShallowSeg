#!/usr/bin/env bash
source activate tensorflow


bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
        --in_graph=./log/quantised/frozen_graph.pb \
        --out_graph=./log/quantised/optimised_graph.pb \
        --inputs=['ShallowSeg/downsampling_block_1_conv','ShallowSeg/downsampling_block_1_pool'] \
        --outputs='ShallowSeg/Softmax' \
        --transforms='
         add_default_attributes
         strip_unused_nodes(type=float, shape="1,224,224,3")
         remove_nodes(op=Identity, op=CheckNumerics)
         fold_constants(ignore_errors=true)
         fold_batch_norms
         fold_old_batch_norms
         quantize_weights
         quantize_nodes
         strip_unused_nodes
         sort_by_execution_order'

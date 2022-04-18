# Converts the MobileNetV2+SSDLite model to ONNX.
#
# This script creates a pipeline with three models:
#   1. MobileNetV2 + SSDLite
#   2. A neural network that decodes the coordinate predictions using the anchor boxes.
#
# This is the model from the paper 'SSD: Single Shot MultiBox Detector' by Liu et al (2015),
# https://arxiv.org/abs/1512.02325, with MobileNetV2 as the backbone and depthwise separable
# convolutions for the SSD layers (also known as SSDLite).
#
# The version of the model used is ssdlite_mobilenet_v2_coco, downloaded from:
# http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
#
# It was originally trained with the TensorFlow Object Detection API:
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
#
# The model expects input images of 300x300 pixels and detects objects from the COCO dataset.
#
# NOTE: The conversion script reads from saved_model.pb, not from frozen_inference_graph.pb.
# (Using the frozen graph gives an error, "ValueError: Graph has cycles".)
#
# Tested with Python 3.6.5, Tensorflow 1.7.0, coremltools 2.0, tfcoreml 0.3.0.
# See also: https://github.com/tf-coreml/tf-coreml/blob/master/examples/ssd_example.ipynb


# system tensorflow 1.15.1 dont use tensorflow2.x
import os
import sys

import numpy as np

import tensorflow as tf
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

import onnx
from tf2onnx.tfonnx import process_tf_graph
from tf2onnx import constants, logging, utils, optimizer
from tf2onnx import tf_loader
from tf2onnx.graph import ExternalTensorStorage
from tf2onnx.tf_utils import compress_graph_def


# From where to load the saved_model.pb file.
saved_model_path = "saved_model"

# Where to save the final Core ML model file.
onnx_model_path = "mobilenetv2-ssdlite.onnx"

# The number of predicted classes, excluding background.
num_classes = 90

# The number of predicted bounding boxes.
num_anchors = 1917

# Size of the expected input image.
input_width = 300
input_height = 300


# Temporary file. You can delete this after the conversion is done.
frozen_model_file = "frozen_model.pb"

# Names of the interesting tensors in the graph. We use "Postprocessor/convert_scores"
# instead of "concat_1" because this already applies the sigmoid to the class scores.
input_node = "Preprocessor/sub"
bbox_output_node = "concat"
class_output_node = "Postprocessor/convert_scores"

input_tensor = input_node + ":0"
bbox_output_tensor = bbox_output_node + ":0"
class_output_tensor = class_output_node + ":0"

input_names=[input_tensor]
output_names=[class_output_tensor, bbox_output_tensor]
inputs_as_nchw=[input_tensor]


def convert2onnx(frozen_graph, name="unknown", large_model=False, output_path=None,
                    output_frozen_graph=None, custom_ops=None, custom_op_handlers=None, **kwargs):
    """Common processing for conversion."""

    model_proto = None
    external_tensor_storage = None
    const_node_values = None

    if custom_ops is not None:
        if custom_op_handlers is None:
            custom_op_handlers = {}
        custom_op_handlers.update(
            {op: (make_default_custom_op_handler(domain), []) for op, domain in custom_ops.items()})

    with tf.Graph().as_default() as tf_graph:
        if large_model:
            const_node_values = compress_graph_def(frozen_graph)
            external_tensor_storage = ExternalTensorStorage()
        if output_frozen_graph:
            utils.save_protobuf(output_frozen_graph, frozen_graph)
        if not kwargs.get("tflite_path") and not kwargs.get("tfjs_path"):
            tf.import_graph_def(frozen_graph, name='')
        g = process_tf_graph(tf_graph, const_node_values=const_node_values,
                             custom_op_handlers=custom_op_handlers, **kwargs)
        if constants.ENV_TF2ONNX_CATCH_ERRORS in os.environ:
            catch_errors = constants.ENV_TF2ONNX_CATCH_ERRORS.upper() == "TRUE"
        else:
            catch_errors = not large_model
        onnx_graph = optimizer.optimize_graph(g, catch_errors)
        model_proto = onnx_graph.make_model("converted from {}".format(name),
                                            external_tensor_storage=external_tensor_storage)
    if output_path:
        if large_model:
            utils.save_onnx_zip(output_path, model_proto, external_tensor_storage)
        else:
            utils.save_protobuf(output_path, model_proto)

    return model_proto, external_tensor_storage


def load_saved_model(path):
    """Loads a saved model into a graph."""
    graph_def, inputs, outputs, initialized_tables, tensors_to_rename = tf_loader.from_saved_model(
            path, input_names, output_names, return_initialized_tables=True, return_tensors_to_rename=True)
    return graph_def


# Load the original graph and remove anything we don't need.
the_graph = load_saved_model(saved_model_path)

# Convert to ONNX model.
onnx_graph, external_tensor_storage = convert2onnx(the_graph, output_path=onnx_model_path,
                                             input_names=[input_tensor], output_names=[class_output_tensor, bbox_output_tensor],
                                             inputs_as_nchw=[input_tensor])

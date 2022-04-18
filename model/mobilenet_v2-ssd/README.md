mobilenetv2_ssd is supported by TNN before 2021.12.23, which is converted from caffe model with priorbox layer.
mobilenetv2_ssd_tf is converted from tensorflow version by ssdlite_tf2onnx.py, TNN supportes it since 2021.12.23. mobilenetv2_ssd_tf has no priorbox layer, the priorbox has been precomputed in the source code as ssd_anchors (detector_utils.h).
mobilenetv2_ssd_tf_fix_box is also converted from tensorflow version with the post porocess for decoding the output boxes. TNN supportes it since 2022.04.18

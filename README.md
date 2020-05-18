# tflite-generation-interpretation

Tflite files are compact version of our frozen graphs designed to run efficienty in limited resource setting like a smartphone or on an edge devices like raspberry pi. These will work best if used with Mobilenet checkpoints, but if you want better accuracy and can work with a bit slower prediction model like Densenet then feel free to use that as your backbone.

Important Note: Tflite files can only be generated for SSD architectures as of April 2020.

You must have trained your ssd model and have finalised checkpoint to be used by now.

Before running the following commands add '/tensorflow/models/research' and '/tensorflow/models/research/slim' to your PYTHONPATH variable and switch to 'tensorflow/models' directory.

Last I ran these commands with Tensorflow 1.14 and Python 3.7

## 1. Tflite generation for Object Detection Checkpoints

python research/object_detection/export_tflite_ssd_graph.py \
  --pipeline_config_path /zeleling/ssdlite_mn_v2/pipeline.config \
  --trained_checkpoint_prefix /zeleling/ssdlite_mn_v2/train/model.ckpt-10187 \
  --output_directory /zeleling/ssdlite_mn_v2/train/tflite \
  --add_postprocessing_op=true

tflite_convert \
  --graph_def_file=/zeleling/ssdlite_mn_v2/train/tflite/tflite_graph.pb \
  --output_file=/zeleling/ssdlite_mn_v2/train/tflite/detect_float.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --input_shapes=1,300,300,3 \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
  --output_format=TFLITE \
  --inference_type=FLOAT \
  --allow_custom_ops \
  --std_dev=127 \
  --mean=128 \
  --default_ranges_min=0 \
  --default_ranges_max=6

tflite_convert \
  --graph_def_file=/zeleling/ssdlite_mn_v2/train/tflite/tflite_graph.pb \
  --output_file=/zeleling/ssdlite_mn_v2/train/tflite/detect_quant8.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --input_shapes=1,300,300,3 \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
  --output_format=TFLITE \
  --inference_type=QUANTIZED_UINT8 \
  --allow_custom_ops \
  --std_dev=127 \
  --mean=128 \
  --default_ranges_min=0 \
  --default_ranges_max=6
  
## 2. Tflite generation for Image Classification Checkpoints

python research/object_detection/export_tflite_ssd_graph.py \
  --pipeline_config_path /zeleling/ssdlite_mn_v2/pipeline.config \
  --trained_checkpoint_prefix /zeleling/ssdlite_mn_v2/train/model.ckpt-10187 \
  --output_directory /zeleling/ssdlite_mn_v2/train/tflite \
  --add_postprocessing_op=true
  
tflite_convert \
  --graph_def_file=/zeleling/ssdlite_mn_v2/train/tflite/tflite_graph.pb \
  --output_file=/zeleling/ssdlite_mn_v2/train/tflite/detect_float.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --input_shapes=1,300,300,3 \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
  --output_format=TFLITE \
  --inference_type=FLOAT \
  --allow_custom_ops \
  --std_dev=127 \
  --mean=128 \
  --default_ranges_min=0 \
  --default_ranges_max=6

tflite_convert \
  --graph_def_file=/zeleling/ssdlite_mn_v2/train/tflite/tflite_graph.pb \
  --output_file=/zeleling/ssdlite_mn_v2/train/tflite/detect_quant8.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --input_shapes=1,300,300,3 \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
  --output_format=TFLITE \
  --inference_type=QUANTIZED_UINT8 \
  --allow_custom_ops \
  --std_dev=127 \
  --mean=128 \
  --default_ranges_min=0 \
  --default_ranges_max=6

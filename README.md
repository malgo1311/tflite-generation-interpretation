# Tflite generation from Tensorflow Graphdef

Tflite files are compact version of our frozen graphs designed to run efficienty in limited resource setting like a smartphone or on an edge devices like raspberry pi. These will work best if used with Mobilenet checkpoints, but if you want better accuracy and can work with a bit slower prediction model like Densenet then feel free to use that as your backbone.

Important Note: Tflite files can only be generated for SSD architectures as of April 2020.

You must have trained your ssd model and have finalised checkpoint to be used by now.

Before running the following commands add '/tensorflow/models/research' and '/tensorflow/models/research/slim' to your PYTHONPATH variable and switch to 'tensorflow/models' directory.

Last I ran these commands with Tensorflow 1.14 and Python 3.7

## Tflite generation for Object Detection Checkpoints

python research/object_detection/export_tflite_ssd_graph.py \
  --pipeline_config_path /zeleling/ssdlite_mn_v2/pipeline.config \
  --trained_checkpoint_prefix /zeleling/ssdlite_mn_v2/train/model.ckpt-10187 \
  --output_directory /zeleling/ssdlite_mn_v2/train/tflite \
  --add_postprocessing_op=true

#### For normally trained models you should set 'inference_type=FLOAT' and run the following command by using your custom 'input_shapes'

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
  
#### If quantized training is performed, then you can set 'inference_type=QUANTIZED_UINT8'. These files are faster than float inference tflites.

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
  --default_ranges_max=6 \
  --change_concat_input_ranges=true
  
Regarding default_ranges_min and default_ranges_max: (From https://www.tensorflow.org/lite/convert/cmdline_examples) The example contains a model using Relu6 activation functions. Therefore, a reasonable guess is that most activation ranges should be contained in [0, 6].

You can learn more about the arguments that we are passing in the command here - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/python/tflite_convert.py
  
## Tflite generation for Image Classification Checkpoints

Most of the stuff mentioned above is true for these checkpoints as well. You just need to replace the following.

--input_arrays=input \
--output_arrays=MobilenetV1/Predictions/Reshape_1

'output_arrays' may change according to the model you are using.

# Tflite Interpretation

1. tflite-intepretation.py - This script can be used to verify if the tflite generated behaves in the way you are expecting in your python environment

2. interpret_using_tflite_runtime.py - I had written this script to predict on a raspberry-pi, and tflite_runtime is a little lighter. We do not need to install tensorflow on our rpi, instead we can just use tflite_runtime library

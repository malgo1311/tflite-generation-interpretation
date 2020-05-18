import tensorflow as tf
import os
import numpy as np
import cv2
from glob import glob
import time
import argparse
from PIL import Image
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--tflite_path", help="echo the string you use here")
parser.add_argument("--py_img_package", help="can be cv2 or pil", default='cv2')
args = parser.parse_args()

print(tf.__version__) # 1.14

def create_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir) 
        os.mkdir(dir)
    else:
        os.mkdir(dir)
    return

tflite_file_path = args.tflite_path

min_conf_threshold = 0.8

label_name_dict = {0: 'green', 1: 'person', 2: 'red'}

# Load TFLite model and allocate tensors.
# interpreter = tf.contrib.lite.Interpreter(model_path=name)
interpreter = tf.lite.Interpreter(model_path=tflite_file_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
print(input_details)

# Get output tensors.
output_details = interpreter.get_output_details()
print(output_details[0])

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 128
input_std = 127

print('Input image height, width', height, width)
print('Model inference type', floating_model)

all_images = glob('test_images/*.jpeg')
print('Total images', len(all_images))

counter = 0
start = time.time()
for image_path in all_images:

    if args.py_img_package=='pil':
        img = Image.open(image_path)
        imH, imW = im.size
        img_resized = img.resize((width, height))
        input_data = np.expand_dims(img_resized, axis=0)

    elif args.py_img_package=='cv2':
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        imH, imW, _ = image.shape 
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)
            
    else:
        print('Please provide correct python image package')
        print('Aborting...')
        break

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
#         print(scores[i])
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            counter += 1

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

end = time.time()

print('Entire execution took (seconds)', end-start)
print('Execution time per image (ms)', (end-start)/len(all_images)*1000)
print('Python package used:', args.py_img_package)
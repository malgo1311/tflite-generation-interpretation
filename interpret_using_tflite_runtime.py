import argparse
from tflite_runtime.interpreter import Interpreter
import numpy as np
from PIL import Image
from glob import glob
import time
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--tflite_path", help="echo the string you use here")
parser.add_argument("--py_img_package", help="can be cv2 or pil", default='cv2')
args = parser.parse_args()

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def tensorflow_lite(tflite_path):

    print(tflite_path)
    interpreter = Interpreter(model_path=tflite_path)
        #     experimental_delegates=[load_delegate('libedgetpu.so.1.0')], #with or without it
        # )
    interpreter.allocate_tensors()
   
    input_details = interpreter.get_input_details()
    print(input_details)

    output_details = interpreter.get_output_details()
    print(output_details)

    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    floating_model = (input_details[0]['dtype'] == np.float32)
    input_mean = 128
    input_std = 127

    inference_type = 'float' if floating_model else 'quant'

    print('Input image height, width', height, width)
    print('Model inference type', inference_type)

    all_images = glob('test_images/*.jpeg')
    print('Total images', len(all_images))

    start = time.time()
    for image_path in all_images:
        
        if args.py_img_package=='pil':
            img = Image.open(image_path).resize((width, height))
            input_data = np.expand_dims(img, axis=0)

        elif args.py_img_package=='cv2':
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_resized = cv2.resize(image_rgb, (width, height))
            input_data = np.expand_dims(image_resized, axis=0)
        else:
            print('Please provide correct python image package')
            return()

        # image = np.zeros((1, height, width, 3,), dtype=np.float32)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        set_input_tensor(interpreter, input_data)
        interpreter.invoke()
        output = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
        
    end = time.time()

    print('Entire execution took (seconds)', end-start)
    print('Execution time for 1 image (ms)', (end-start)/len(all_images)*1000)
    print('Python package used:', args.py_img_package)

def main():
    tensorflow_lite(args.tflite_path)

if __name__ == '__main__':
    main()
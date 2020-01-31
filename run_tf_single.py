#import tensorflow as tf
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
from skimage.io import imread, imsave
import sys

def label_prediction(labelarray):
#  print(labelarray.shape)
  for i in labelarray:
    return np.argmax(i)

def LABELS(index):
  labels = ['air', 'aut', 'brd', 'cat', 'dir', 'dog', 'frg',
          'hrs', 'shp', 'trk']
  return labels[index]

  
def run(folder,imagenum,imagelabel,advlabel):
  print(imagenum)
  print(imagelabel)
  print(advlabel)

  # Load TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path="converted_model_cifar.tflite")
  interpreter.allocate_tensors()
  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  # input output index
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Test model on random input data.
  input_shape = input_details[0]['shape']
  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()

  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  output_data = interpreter.get_tensor(output_details[0]['index'])


  #get image numpy array
  impath = "%s/%sa/%s_modified_into_%s.png"%(folder,imagenum,imagelabel,advlabel)
  x = imread(impath)
  x = x/255
  x = np.transpose(x,(2,0,1))
  x_expand = np.empty([1,3072], dtype=np.float32)
  x_expand[0] = np.reshape(x,3072)
  interpreter.set_tensor(input_index,x_expand)
  interpreter.invoke()
  predictions = interpreter.get_tensor(output_index)
  label = LABELS(label_prediction(predictions))

  if label == imagelabel:
    robust="T"
  elif label == advlabel:
    robust = "F"
  else:
    robust = "O"

  f = open(sys.argv[5], "a+")
  wr="%s,%s,%s,%s,%s \n"%(imagenum,imagelabel,advlabel,label,robust)
  print(wr)
  f.write(wr)
#  print("image index:", imagenum)
#  print("folder size:", foldersize)
#  print("num correct: ", num_correct)
#  print("num seen: ", num_seen)

def main():
  run(str(sys.argv[1]),int(sys.argv[2]),str(sys.argv[3]),str(sys.argv[4]))

if __name__ == "__main__":

    main()

# TensBlur: utilities to blur images or layers in tensorflow

## Dependencies

1. Numpy
2. Scipy
3. Tensorflow
4. Pillow

## Class Smoother

The file `smoother.py` contains the main class to implement the blurring on the feeded input. 
This input can be either an image (as shown in the example) or one of the layers of a Neural Network.
The main class to import is `Smoother`. The class is a forked version of [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow)

## Run

To run the example:
`python2.7 blur_image.py`

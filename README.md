# Emotion Detector

A small 2021 project I did to learn how CNNs work. It reads your webcam, finds faces with OpenCV's Haar cascade, and guesses whether each one looks angry, happy, neutral, sad or surprised, using a small convolutional neural network trained with Keras.

## Files

`Classification_little_vgg.py` trains the model on 48×48 grayscale face images. The network is a cut-down VGG: four convolution blocks (32, 64, 128 and 256 filters) with batch normalisation, max pooling and dropout, then two dense layers. The training images are randomly rotated, shifted, sheared, zoomed and flipped using `ImageDataGenerator`. Training stops early when it stops improving, and the best model is saved to `Emotion_little_vgg.h5`.

`Facial_Expressions_Recog.py` is the live part. It takes frames from the webcam, finds faces, shrinks each one to 48×48, and writes the predicted emotion on the video.

## Running it

```sh
pip install tensorflow opencv-python numpy
python Facial_Expressions_Recog.py   # uses the model saved in the repo, press q to quit
```

The dataset isn't in the repo. To train again, put 48×48 grayscale images (FER-2013 style) in `train/<emotion>/` and `validation/<emotion>/`. I used about 24,000 for training and 3,000 for validation.

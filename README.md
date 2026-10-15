# Emotion Detector

Real-time facial expression recognition from a webcam, using a small convolutional neural network (CNN) trained with Keras.

It detects faces with OpenCV's Haar cascade, then classifies each face as **Angry, Happy, Neutral, Sad or Surprise**.

## How it works

1. **Training** (`Classification_little_vgg.py`): a compact VGG-style CNN ("mini-VGG") trained on 48×48 grayscale face images.
   - Four convolutional blocks (32 → 64 → 128 → 256 filters) with batch normalisation, max pooling and dropout, followed by two dense layers.
   - Data augmentation (rotation, shifts, shear, zoom, horizontal flip) with `ImageDataGenerator`.
   - `ModelCheckpoint`, `EarlyStopping` and `ReduceLROnPlateau` callbacks; the best model is saved to `Emotion_little_vgg.h5`.
2. **Live detection** (`Facial_Expressions_Recog.py`): reads frames from the webcam, finds faces, resizes each to 48×48, and draws the predicted emotion on screen.

## Running it

```sh
pip install tensorflow opencv-python numpy
python Facial_Expressions_Recog.py   # uses the trained model included in the repo; press q to quit
```

To retrain, put the images in `train/<emotion>/` and `validation/<emotion>/` folders (about 24,000 and 3,000 images in my run; 48×48 grayscale, FER-2013 format). The dataset isn't stored in this repository.

## Notes

A small 2021 prototype to learn how CNNs work end to end, from data augmentation and training callbacks to running inference on a live video feed.

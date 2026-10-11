# LEARNING CNN

Facial emotion detection is in [`emotionDetection/`](emotionDetection/).

A convolutional network trains on grayscale face photos in seven emotion folders. The webcam script finds a face and labels the expression in real time.

## Run the camera

```powershell
py emotionDetection/emotionDetection.py
```

Press `q` to close the window.

## Train again

```powershell
py emotionDetection/emotionDetection.py --train
```

That fits the Keras model for 30 epochs and saves `emotion_model.weights.h5`. The live camera uses the ONNX models in `emotionDetection/models/`..

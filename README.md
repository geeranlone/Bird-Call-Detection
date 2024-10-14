# Bird Call Detection
This project provides a neural network-based solution to detect Capuchin bird calls in audio files. It processes .mp3 files, extracts features in the form of spectrograms, and classifies the presence of Capuchin bird calls using a Convolutional Neural Network (CNN) model. This repository includes code for both training and inference, as well as an interactive Jupyter notebook frontend for users to upload audio files and obtain predictions.


# Overview
The main goal of this project is to detect Capuchin bird calls from forest recordings. The CNN model takes audio files as input, preprocesses them into spectrograms, and predicts whether the audio contains Capuchin bird calls. A Jupyter notebook frontend using ipywidgets allows users to interact with the trained model by uploading audio files and getting real-time predictions.

# Features
Preprocessing of audio files (WAV/MP3) to 16 kHz mono.
CNN-based binary classification to detect Capuchinbird calls.
Interactive Jupyter notebook frontend using ipywidgets.
Audio file upload and real-time predictions of Capuchinbird calls.
Visual display of the number of Capuchinbird calls detected in the audio.
# Setup
To run this project, you'll need to set up a Python environment with the required dependencies. Follow the steps below:


Prerequisites


Python 3.8 or higher


Jupyter Notebook


TensorFlow 2.x


# Installation
Clone the repository:

```
git clone https://github.com/your-username/capuchinbird-call-detection.git
cd capuchinbird-call-detection
```
Install the required dependencies:


Set up Jupyter Notebook:

```
pip install notebook
jupyter nbextension enable --py widgetsnbextension
```
# Dataset
```
https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing
```


# Model Training
If you want to retrain the model from scratch:

Organize your dataset of Capuchinbird audio files (.wav format) and negative samples (non-Capuchinbird audio).

Define the paths to your dataset in the notebook or Python script.

Use the load_wav_16k_mono function to preprocess audio files into 16kHz mono format.

Preprocess the audio into spectrograms and build a training dataset.

Define the CNN model using tensorflow.keras and train it using your dataset.

Save the trained model using:

```
model.save('capuchin_model.h5')
```
# Model
The model used in this project is a simple Convolutional Neural Network (CNN) consisting of:



2D Convolutional layers for extracting features from spectrograms.


Global Average Pooling for dimensionality reduction.


Fully connected layers for binary classification.


Sigmoid activation for binary output (presence of Capuchinbird calls).


The model is trained on spectrograms generated from audio clips of Capuchinbird calls and non-Capuchinbird sounds.

# Results
The notebook provides:



Predictions on uploaded audio files.


Number of Capuchinbird calls detected.


Visualization of the audio waveform and spectrogram (optional, via matplotlib).


# Contributing
Contributions are welcome! If you would like to contribute to the project, feel free to open an issue or submit a pull request.

# Fork the repository.
Create your feature branch 
```
(git checkout -b feature/AmazingFeature).
```
Commit your changes 
```
(git commit -m 'Add some AmazingFeature').
```
Push to the branch
```
(git push origin feature/AmazingFeature).
```
Open a pull request.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

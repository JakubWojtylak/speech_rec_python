
import os
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import warnings
import pathlib
import time
import sounddevice as sd
import soundfile as sf
from keras.utils import np_utils
from keras.models import load_model

import tensorflow as tf
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

curr_path = pathlib.Path(__file__).parent.absolute()
print(curr_path)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
model = load_model('best_model.hdf5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

lite_model_name = "best_model_lite.hdf5"
open(lite_model_name,"wb").write(tflite_model)
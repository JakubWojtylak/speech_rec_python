

import os
import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
from scipy.io import wavfile
import warnings
import pathlib
import time
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def predict(audio):
    prob = model.predict(audio.reshape(1,8000,1))
    index = np.argmax(prob[0])
    return classes[index]


def read_audio():
    samplerate = 16000
    duration = 1  # seconds
    filename = 'test.wav'
    print("speak in 1...")
    time.sleep(1)
    print("start of recording")
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
                    channels=1, blocking=True)
    print("end of recording")
    sd.wait()
    sf.write(filename, mydata, samplerate)

def test_audio(filename,classes):
    samples, sample_rate = librosa.load(filename, sr=16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    ipd.Audio(samples, rate=8000)
    prob = model.predict(samples.reshape(1, 8000, 1))
    index = np.argmax(prob[0])
    print("you said: " + classes[index])

def test_audio_with_tflite_model(filename,classes):
    samples, sample_rate = librosa.load(filename, sr=16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    ipd.Audio(samples, rate=8000)
    interpreter = tf.lite.Interpreter(model_path="best_model_lite.tflite")
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(samples.reshape(1, 8000, 1), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    index = np.argmax(output_data[0])
    print("tflite you said: " + classes[index])



if __name__ == "__main__":

    labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    print("POSSIBLE COMMANDS ARE: " , labels)
    print("loading model...")


    classes2 = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
    from keras.utils import np_utils
    from keras.models import load_model
    model = load_model('best_model.hdf5')
    filepath = pathlib.Path(__file__).parent.absolute()
    file_name = str(filepath) + '/test.wav'
    choice = 0
    end_program = False
    correct = 0
    total = 0
    for i in range(100):
        if end_program:
            break
        read_audio()
        test_audio(file_name, classes2)
        test_audio_with_tflite_model(file_name, classes2)
        total += 1
        choice = input("do you want to finish?? y/n ")
        if choice == 'y':
           end_program = True
        elif choice == 'n':
            continue


    print("acc: ", float(correct)/float(total))
    print("TERMINATING..")


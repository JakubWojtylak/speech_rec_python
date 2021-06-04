

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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def predict(audio):
    prob = model.predict(audio.reshape(1,8000,1))
    index = np.argmax(prob[0])
    return classes[index]


def read_audio():
    samplerate = 16000
    duration = 1  # seconds
    filename = 'test.wav'
    print("speak in 3...")
    time.sleep(1)
    print("speak in 2...")
    time.sleep(1)
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

if __name__ == "__main__":

    labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    print("POSSIBLE COMMANDS ARE: " , labels)
    print("loading model...")
    # all_wave = []
    # all_label = []
    # filepath = pathlib.Path(__file__).parent.absolute()
    # train_audio_path = str(filepath) + '/tensorflow-speech-recognition-challenge/train/audio/'
    #
    # for label in labels:
    #     waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
    #     for wav in waves:
    #             all_label.append(label)
    #
    # from sklearn.preprocessing import LabelEncoder
    #
    # le = LabelEncoder()
    # y = le.fit_transform(all_label)
    # classes = list(le.classes_)
    # print(classes)

    classes2 = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
    from keras.utils import np_utils
    from keras.models import load_model
    model = load_model('best_model.hdf5')
    filepath = pathlib.Path(__file__).parent.absolute()
    file_name = str(filepath) + '/test.wav'
    choice = 0
    end_program = False

    while True:
        if end_program:
            break
        read_audio()
        test_audio(file_name, classes2)
        while True:
            choice = input("do you want to continue? y/n ")
            if choice == 'y':
                break
            elif choice == 'n':
                end_program = True
                break
            else:
                print("wrong answer. Enter y/n ")
                continue



    print("TERMINATING..")


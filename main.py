import os
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import warnings
import pathlib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

curr_path = pathlib.Path(__file__).parent.absolute()
print(curr_path)
warnings.filterwarnings("ignore")

train_audio_path = str(curr_path) + '/tensorflow-speech-recognition-challenge/train/audio/'
samples, sample_rate = librosa.load(train_audio_path + 'yes/0a7c2a8d_nohash_0.wav', sr = 16000)
# fig = plt.figure(figsize=(14, 8))
# ax1 = fig.add_subplot(211)
# ax1.set_title('Raw wave of ' + '../input/train/audio/yes/0a7c2a8d_nohash_0.wav')
# ax1.set_xlabel('time')
# ax1.set_ylabel('Amplitude')
# ax1.plot(np.linspace(0,sample_rate/len(samples), sample_rate), samples)

# plt.show()

ipd.Audio(samples, rate = sample_rate)
print(sample_rate)

# labels = os.listdir(train_audio_path)
# no_of_recordings=[]
# for label in labels:
#     waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
#     no_of_recordings.append(len(waves))
#
#
# plt.figure(figsize=(20, 5))
# index = np.arange(len(labels))
# plt.bar(index, no_of_recordings)
# plt.xlabel('Commands', fontsize=12)
# plt.ylabel('No of recordings', fontsize=12)
# plt.xticks(index, labels, fontsize=15, rotation=60)
# plt.title('No. of recording for each command')
# # plt.show()

labels=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

# duration_of_recordings = []
# for label in labels:
#     waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
#     for wav in waves:
#         sample_rate, samples = wavfile.read(train_audio_path + '/' + label + '/' + wav)
#         duration_of_recordings.append(float(len(samples) / sample_rate))
#
# plt.hist(np.array(duration_of_recordings))
all_wave = []
all_label = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if(len(samples)== 8000) :
            all_wave.append(samples)
            all_label.append(label)

# plt.hist(np.array(duration_of_recordings))
# plt.show()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(all_label)
classes = list(le.classes_)

from keras.utils import  np_utils
y = np_utils.to_categorical(y, num_classes=len(labels))

all_wave = np.array(all_wave).reshape(-1, 8000, 1)

from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave), np.array(y), stratify = y, test_size=0.2, \
                                            random_state=777,shuffle = True)


from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.clear_session()

inputs = Input(shape=(8000, 1))

#first conv1D layer
conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#second Conv1D layes
conv = Conv1D(16,11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)


#Third Conv1D layer
conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Fourth Conv1D layer
conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Flatten layer
conv = Flatten()(conv)

#Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

#Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(labels), activation='softmax')(conv)

model = Model(inputs, outputs)
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10,min_delta=0.0001)
mc = ModelCheckpoint(str(curr_path) + '/best_model2.hdf5', monitor='val_acc', verbose=1, save_best_only=False, mode='max')
history = model.fit(x_tr, y_tr, epochs=100, callbacks=[es,mc], batch_size=32, validation_data=(x_val, y_val))


from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# pyplot.show()

print("end")
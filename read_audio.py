import sounddevice as sd
import soundfile as sf

def read_audio():
    samplerate = 16000
    duration = 2  # seconds
    filename = 'down.wav'
    print("start")
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
                    channels=1, blocking=True)
    print("end")
    sd.wait()
    sf.write(filename, mydata, samplerate)
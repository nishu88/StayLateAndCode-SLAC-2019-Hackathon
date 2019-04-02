from flask import Flask, render_template
    #ONLY PREDICT DO NOT RECORD

import pyaudio
import wave
import os
import librosa
import numpy as np
from hmmlearn.hmm import GMMHMM
import pickle
from sklearn.externals import joblib
import pandas as pd
import speech_recognition as sr
import re
from os import path
import sys

import pyaudio
import wave


import random
import librosa.display
import matplotlib.pyplot as plt
import cv2


__author__ = 'Freefall'

app = Flask(__name__)



@app.route('/')
def index():
        return render_template('index.html')


@app.route('/hi')
def final():
        def record():


    
            #RECORD SPEAKERS VOICE FOR GIVING AUTHENTICATION
            name="1"    
            FORMAT = pyaudio.paInt16
            CHANNELS = 2
            RATE = 44100
            CHUNK = 1024
            RECORD_SECONDS = 3
            WAVE_OUTPUT_FILENAME = r"C:\Users\nisha\Desktop\SLAC\Authentication/samples/"+name+".wav"
             
            audio = pyaudio.PyAudio()
             
            # start Recording
            stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
            print("recording...", file=sys.stderr)
            frames = []
             
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            
            print("finished recording", file=sys.stderr)
             
             # stop Recording
            stream.stop_stream()
            stream.close()
            audio.terminate()
             
            waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()

            

        #speakers =["nishant","padma","rajat","shreekar","shruthi"]

        with open(r'C:\Users\nisha\Desktop\SLAC\Authentication/s.txt') as f:
                
            speakers = f.read().splitlines()
            speakers=speakers[:len(speakers)]

    #speakers1 = [line.strip() for line in open('C:/Anaconda codes/speaker reco/something new/for hack/s.txt', 'r')]
        print((speakers), file=sys.stderr)

#speakers =["nishant","padma","rajat","shreekar","shruthi"]
#print(speakers,"\n")


    #f=input("enter name to predict")

        record()

    #speakers =["nishant","padma","rajat","rohit","sarah","shreekar","shruthi"]


        threshold = 100 
        l=2
        uppercutoff=20000
        lowercutoff=8000

            #open the test data and find its probability 
            #compare it with test probability and print predictions

        student="samples"+f+".wav" #SPEAKERS VOICE STORED

        file_path1=r"C:\Users\nisha\Desktop\SLAC\Authentication/"+student
        file_path1=file_path1.decode('utf-8')
            #file_path1="C:/Anaconda codes/speaker reco/something new/for hack/other students/"
        test_speech1 = student
        
        speech1, rate = librosa.core.load(file_path1)     #EXTRACT MFCC AND ADD IT OT FEATURE VECTOR
        feature_vectors12 = librosa.feature.mfcc(y=speech1, sr=rate)

        features1=feature_vectors12.transpose()
    #print(np.shape(features1))





    #GET THE PREDICTION VALUES FOR EVERY MODEL CREATED FOR EACH SPEAKER
        x=[]

        path =r"C:\Users\nisha\Desktop\SLAC\Authentication/models/"
        names = os.listdir(path)
        print(names, file=sys.stderr)

        h=[]
        for i in range(0,len(names)):
                
                m1=joblib.load(r"C:\Users\nisha\Desktop\SLAC\Authentication/models/"+str(names[i]) )
                p1 = m1.score(features1)
                p1=abs(p1)
                x.append(p1)
                print(m1.predict(features1),'\n')



        y=x.index(min(x))
        print(x)
            #print(x.index(min(x))+1)
        if min(x)<uppercutoff and min(x)>lowercutoff:
                
                print("Hi "+speakers[y]+".How are you?", file=sys.stderr)
                p=speakers[y]
                
        else:
                print("cant recognise. Speak again", file=sys.stderr)
                

        return render_template('output.html') 


        final()



if __name__ == '__main__':
    
    app.run()

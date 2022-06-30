import numpy as np
from keras.models import load_model
import os


classes=['Alif', 'Ba', 'Jeem']


def predict(audio):
    audio=audio[0:23552]
    # put the path of model on your PC
    model = load_model("my_model2")
    prob=model.predict(audio.reshape(1,23552,1))  #prediction will enter the audio to layers of model so it will enter to i/p layer
    #which takes 23552 as input shape
    index=np.argmax(prob[0])
    return classes[index]



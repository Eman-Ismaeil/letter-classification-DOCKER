import os
import librosa   #for audio processing
import IPython.display as ipd
from keras.models import load_model
from flask import Flask, request, jsonify
from prediction_postman import predict

app = Flask(__name__)

@app.route('/', methods=["POST"])
def output():
    if request.method=="POST":
        file = request.files['file']
        samples, sample_rate = librosa.load(file, sr=16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        ipd.Audio(samples,rate=8000)
    return jsonify({'letter' :predict(samples)})



app.run(host='0.0.0.0', port='5000')




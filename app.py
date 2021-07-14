import os
from tensorflow import keras
from flask import Flask, request, render_template, flash, redirect
#from keras.models import load_model
import pickle
import librosa
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
import tensorflow as tf
import utils
from utils.feature_extraction import get_audio_features
from utils.feature_extraction import get_features_dataframe


app = Flask(__name__)

model=tf.keras.models.load_model('model')
opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
emotions={0:"anger",1:"disgust",2:"fear",3:"happy",4:"neutral",5:"sad",6:"surprise"}


@app.route("/",methods=['GET','POST'])
def index():
  return render_template('index.html')


@app.route("/predict", methods=['GET','POST'])
def predict():
  if request.method == 'POST':
        file = request.files['file']
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
          basepath=os.path.dirname(__file__)
          filepath=os.path.join(basepath,'uploads/',secure_filename(file.filename))
          file.save(filepath)
          demo_mfcc, demo_pitch, demo_mag, demo_chrom = get_audio_features(filepath,22050)
          mfcc = pd.Series(demo_mfcc)
          pit = pd.Series(demo_pitch)
          mag = pd.Series(demo_mag)
          C = pd.Series(demo_chrom)
          demo_audio_features = pd.concat([mfcc,pit,mag,C],ignore_index=True)
          demo_audio_features= np.expand_dims(demo_audio_features, axis=0)
          demo_audio_features= np.expand_dims(demo_audio_features, axis=2)
          livepreds = model.predict(demo_audio_features, batch_size=64, verbose=1)
          index = livepreds.argmax(axis=1).item()
          res=emotions[index].upper()
          return render_template('index.html', prediction_text="The predicted emotion is : "+str(res))
  return redirect(request.url)


if __name__ == "__main__":
    app.run(debug=True)

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
##import utils
##from utils.feature_extraction import get_audio_features
##from utils.feature_extraction import get_features_dataframe


def get_audio_features(audio_path,sampling_rate):
    X, sample_rate = librosa.load(audio_path ,res_type='kaiser_fast',duration=2.5,sr=sampling_rate*2,offset=0.5)
    sample_rate = np.array(sample_rate)

    y_harmonic, y_percussive = librosa.effects.hpss(X)
    pitches, magnitudes = librosa.core.pitch.piptrack(y=X, sr=sample_rate)

    mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=13),axis=1)

    pitches = np.trim_zeros(np.mean(pitches,axis=1))[:20]

    magnitudes = np.trim_zeros(np.mean(magnitudes,axis=1))[:20]

    C = np.mean(librosa.feature.chroma_cqt(y=y_harmonic, sr=sampling_rate),axis=1)
    
    return [mfccs, pitches, magnitudes, C]
    
    

def get_features_dataframe(dataframe, sampling_rate):
    labels = pd.DataFrame(dataframe['label'])
    
    features  = pd.DataFrame(columns=['mfcc','pitches','magnitudes','C'])
    for index, audio_path in enumerate(dataframe['path']):
        features.loc[index] = get_audio_features(audio_path, sampling_rate)
    
    mfcc = features.mfcc.apply(pd.Series)
    pit = features.pitches.apply(pd.Series)
    mag = features.magnitudes.apply(pd.Series)
    C = features.C.apply(pd.Series)
    
    combined_features = pd.concat([mfcc,pit,mag,C],axis=1,ignore_index=True)

    return combined_features, labels 


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

import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from featureextraction import extract_features
#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import time
import math

#path to training data
source   = "SampleData/"

#path where training speakers will be saved
modelpath = "speaker_models/"

gmm_files = [os.path.join(modelpath,fname) for fname in
              os.listdir(modelpath) if fname.endswith('.gmm')]

#Load the Gaussian gender Models
models = [pickle.load(open(fname,'rb')) for fname in gmm_files]
speakers = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]

error = 0
total_sample = 0.0

print ("Do you want to Test a Single Audio: Press '1' or The complete Test Audio Sample: Press '0' ?")
take = int(input().strip())
if take == 1:
    print ("Enter the File name from Test Audio Sample Collection :")
    path = input().strip()
    print ("Testing Audio : ", path)
    sr,audio = read(path)
    vector   = extract_features(audio,sr)
    log_likelihood = np.zeros(len(models))

    for i in range(len(models)):
        gmm    = models[i]  #checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    print ("\tdetected as - ", speakers[winner])

    time.sleep(1.0)
elif take == 0:
    print ("Enter the path to samples:")
    path = input().strip()

    speakers = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]

    for speaker in speakers:
        files = [f for f in os.listdir(os.path.join(path, speaker)) if os.path.isfile(os.path.join(path, speaker, f)) & f.endswith(".wav") ]

        # Read the test directory and get the list of test audio files
        for file in files:
                total_sample += 1.0
                path = path.strip()
                print ("Testing Audio : ", speaker, file)
                sr,audio = read(os.path.join(path,speaker,file))
                vector   = extract_features(audio,sr)
                log_likelihood = np.zeros(len(models))

                for i in range(len(models)):
                    gmm = models[i]
                    scores = np.array(gmm.score(vector))
                    log_likelihood[i] = scores.sum()

                winner = np.argmax(log_likelihood)
                print ("\tdetected as - ", speakers[winner])

                checker_name = speaker#path.split("_")[0]

                if speakers[winner] != checker_name:
                    error += 1
                    time.sleep(1.0)

    print(error,"errors out of",total_sample, "samples.")
    accuracy = ((total_sample - error) / total_sample) * 100
    print ("The Accuracy Percentage for the current testing Performance with MFCC + GMM is : ", accuracy, "%")

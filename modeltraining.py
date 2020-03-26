import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn import mixture
from featureextraction import extract_features
import os

#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")

#path to training data
# source   = "development_set/"
source   = "/media/sf_D_DRIVE/voxforge_16_16/training/"

#path where training speakers will be saved

# dest = "speaker_models/"
# train_file = "development_set_enroll.txt"

dest = "speaker_models/"

count = 1
# Extracting features for each speaker (5 files per speakers)
speakers = [ name for name in os.listdir(source) if os.path.isdir(os.path.join(source, name)) ]
print(len(speakers),"speakers identified.")
print(speakers)

features = np.asarray(())

for speaker in speakers:
    files = [f for f in os.listdir(os.path.join(source, speaker)) if os.path.isfile(os.path.join(source, speaker, f)) & f.endswith(".wav") ]
    print(files)
    total_samples = len(files)
    for n, file in enumerate(files):
        # read the audio
        sr,audio = read(os.path.join(source, speaker, file))

        # extract 40 dimensional MFCC & delta MFCC features
        vector   = extract_features(audio,sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))
        # when features of 5 files of speaker are concatenated, then do model training
    	# -> if count == 5: --> edited below
        if n == total_samples-1:
            gmm = mixture.GaussianMixture(n_components = total_samples, covariance_type='diag')
            gmm.fit(features)

            # dumping the trained gaussian model
            picklefile = speaker.split("-")[0]+".gmm"

            pickle.dump(gmm,open(dest + picklefile,'wb'))
            print('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)
            features = np.asarray(())

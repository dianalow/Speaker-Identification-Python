## Modifications to original:
1. I have updated the original code to work with Python 3, and cleaned up directories.
2. Code will now look for top level wav files in subdirectories (names taken as labels) given a specified starting directory.
3. It will automatically use all wav files provided to train the model.
4. I have used the VoxForge dataset for testing.

# Speaker-Identification-System using Python (3.x)
Audio information plays a rather important role in the increasing digital content that is available today; resulting in a need for methodologies that automatically analyze such content. Speaker Identification is one of the vital field of research based upon Voice Signals. Its other notable fields are: Speech Recognition, Speech-to-Text Conversion, and vice versa, etc. 

**Mel Frequency Cepstral Coefficient (MFCC)** is considered a key factor in performing Speaker Identification. But, there are other features lists available as an alternate to MFCC; like- Linear Predictor Coefficient (LPC), Spectrum Sub-band Centroid (SSC), Rhythm, Turbulence, Line Spectral Frequency (LPF), ChromaFactor, etc. 

**Gaussian Mixture Model (GMM)** is the most popular model for training on our data. The training task can also be executed on other significant models; viz. Hidden Markov Model (HMM). Recently, most of the model training phase for a speaker identification project is executed using Deep learning; especially, Artificial Neural Networks (ANN). In this project, we are mainly focused on implementing MFCC and GMM in pair to achieve our target.
We have considered MFCC with “tuned parameters” as the primary feature and delta- MFCC as secondary feature. And, we have implemented GMM with some tuned parameters to train our model. We have performed this project on **two different kinds of Dataset**; viz. 

**“VoxForge” Dataset** and a **custom dataset** which we have prepared by ourselves. We have obtained an outstanding result on both of these Datasets; viz. **100% accuracy on VoxForge Dataset** and **95.29 % accuracy on self prepared Dataset**. We demonstrate that speaker identification task can be performed using MFCC and GMM together with outstanding accuracy in Identification/ Diarization results.

# Methodology used:
1.) Noise reduction and Silence Removal - Audacity Software

2.) Feature Extraction - featureextraction.py ( library - python_speech_features)

3.) Model Training - modeltraining.py ( GMM )

4.) testing - test.py


# Documentation:
Documentation file and screenshots in **documentation/** folder.

# Repo Description:
**speaker_models/** - consists .gmm model trained files. 1 file for each speaker.

# Credits:
Abhijeet Kumar, Spoken Speaker Identification, MachineLearninginAction, Appliedmachinelearning.

# Disclaimer:
Any discrepancy or violation of copyright issues is just a mistake. Big Apologies for that. In such case please contact to me; I will remove that one. In case you copy any content of this repository, You are expected to mention its owner's ( MY) name with little description. 

To explore the project in detail and find its hidden finctionalities , run and explore it by yourself.

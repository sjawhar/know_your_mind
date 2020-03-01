# Adaptive EEG Classification
The reusable codes and datasets for paper: Adaptive Brain Signal Classification with Reinforced Attentive Convolutional Neural Networks.

1. The Dueling DQN codes including:
    1. Environment code: environment.py, which is the most valuable part. Please design your environment based on your classification task. It can not only used on EEG data classification but can be used for any policy optimization.
    2. Agent: agent.py. 
    3. Main: runner.py

2. In this paper we regard the combination of the convolutional mapping procedure and the classifier as the reward model. To make it more readable and help to design your own reward model, we separately write the reward model as a single python file: reward.py. Please design your own reward model corresponding to your problem and load it in runner.py.

## Datasets
1. The eegmmidb dataset can be downloaded in this link:
https://physionet.org/content/eegmmidb/1.0.0/

2. The emotiv is our local dataset, please refer to our PerCom 2018 paper: Converting Your Thoughts to Texts: Enabling Brain Typing via Deep Feature Learning of EEG Signals.
The download link is: 
https://drive.google.com/open?id=0B9MuJb6Xx2PIM0otakxuVHpkWkk

3. The EEG-S is a subset of the eegmmidb dataset. Take one single task and mark the person ID as label. Classifying the person ID, which is so called person identification.

4. The TUH is an excellent public EEG dataset released by Temple University Hospotial. Download here: https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
In this paper, we are using the TUH EEG Seizure Corpus (v1.0.3) released on 26/04/2017. This dataset is 11.6 G, which is too big that we only use a subset of it in our paper.

For more details, please contact 1015xzhang@gmail.com

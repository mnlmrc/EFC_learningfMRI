import os
from pathlib import Path
import numpy as np
import time
import sys
import itertools

baseDir = "/cifs/diedrichsen/data/Chord_exp"
if not os.path.exists(baseDir):
     baseDir = "/Volumes/diedrichsen_data$/data/Chord_exp"

natDir = 'EFC_learningfMRI/natural'
imagingDir = 'EFC_learningfMRI/imaging_data'
anatDir = 'EFC_learningfMRI/anatomicals'
chordDir = 'EFC_learningfMRI/chords'
behavDir = 'EFC_learningfMRI/behavioural'
glmDir = 'EFC_learningfMRI/glm'
roiDir = 'EFC_learningfMRI/ROI'
rdmDir = 'EFC_learningfMRI/rdm'
surfDir = 'EFC_learningfMRI/surfaceWB'
pcmDir = 'EFC_learningfMRI/pcm'
arbuckleDir = 'EFC_learningfMRI/EF_Arbuckle_JNS2020'

naturalDir = 'natural'

emgDir = 'EFC_learningEMG'

atlasDir = 'atlases' #next((Dir for Dir in atlasDirs if Path(Dir).exists()), None)

figDir='/Users/jdiedrichsen/Dropbox/Grants/CIHR_chords_2026/Figures'
if not os.path.exists(figDir):
    figDir = 'paper/figures/'

print(f'base directory found: {baseDir}')
print(f"Atlas directory found: {atlasDir}")
print(f'figure directory found: {figDir}')

fthresh = 1.2  # threshold to exit the baseline area
ftarget = 2
fGain = np.array([1, 1, 1, 1.5, 1.5])
plan_time = 1.
hold_time = .6  # in seconds

fsample = {
    'emg': 2148,
    'force': 500,
}

nblocks = 8

participants = [101, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 113, 114, 116, 117, 118,] # 119]

chordID = np.sort(np.array([21911, 92122, 91211, 22911, 21291, 12129, 12291, 11911]))

# per-finger digit -> letter: 1 extension, 2 flexion, 9 no press
chord_digit_letter = {'1': 'E', '2': 'F', '9': 'N'}
chordLabel = [''.join(chord_digit_letter[d] for d in str(c)) for c in chordID]

# all 70 ways to split the 8 chords into trained (first 4) and untrained (last 4);
# each entry is an 8-element list, ordered like util.get_trained_and_untrained()
chordID_combinations = [
    [int(c) for c in trained] + [int(c) for c in chordID if c not in trained]
    for trained in itertools.combinations(chordID, 4)
]

fingers = ['thumb', 'index', 'middle', 'ring', 'pinkie']

sessions = [3, 9, 23]
nSess = 3
nRuns = 10  # nominal runs per session -- the offset between sessions in the events files' BN

diffCols = [18, 19, 20, 21, 22]
wait_exec = 4

trialPoint_mapping = {
    1: 'success',
    0: 'unsuccess',
}

sess_mapping = {
    'sess03': 1,
    'sess09': 2,
    'sess23': 3
}

TR = 1
nTR = 336

borders = {'L': os.path.join(atlasDir, 'fs_LR.32k.L.border'),
           'R': os.path.join(atlasDir, 'fs_LR.32k.L.border')}

Hem = ['L', 'R']
struct_cortex = ['CortexLeft', 'CortexRight']
rois = {
    'Desikan': [
        'rostralmiddlefrontal',
        'caudalmiddlefrontal',
        'precentral',
        'postcentral',
        'superiorparietal',
        'pericalcarine'
    ],
    'BA_handArea': [
        'ba4a', 'ba4p', 'ba3A', 'ba3B', 'ba1', 'ba2'
    ],
    'ROI': [
        'SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp'
    ],
    'ROI_grouped': [
        'premotor', 'M1-S1', 'parietal', 'V1'
    ]
}

hrf_params = ['delay_response', 'delay_undershoot', 'dispersion_response', 'dispersion_undershoot', 'ratio', 'onset', 'length',]
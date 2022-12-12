
# @author: Franck Porteous <franck.porteous@protonmail.com>
# Code writen for the FINS project 
# Here, we 
#   - Load the dual EEG files corresponding to the last step of
#     preprocessing (rej3, or rej2)
#   - Load the EEG file pairs 
#       - Verifies epoch count (WE HAVE A PROBLEM HERE,
#         working on realigne-eeg-streams.ipynb for a solution)
#   TBC (codes' already there for further steps)


#%% IMPORTS
# FOOOF imports
from copy import deepcopy

from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectrum

import hypyp

import mne
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
import seaborn as sns
from utils.useful_func import *

data_path    = "../FINS-data/"
save_path    = "../FINS-results/"

#%%
# %matplotlib qt 
%matplotlib inline 

#%%
all_dir = os.listdir(data_path) 
all_dir.remove('.DS_Store')

# Get unique dyad number
dyad_nb=[]
for i in all_dir:
    dyad_nb.append(i.split('_')[0])
dyad_nb = np.unique(dyad_nb)


# Extract alpha peaks
# keep which channels are being remove



# %%
valid_chi_dynb_rej2, valid_adu_dynb_rej2, valid_chi_dynb_rej3, valid_adu_dynb_rej3 = [], [], [], []

for dyad in dyad_nb: # for all dyad folder in the main folder
    chi_files = os.listdir('{}{}_child_FP'.format(data_path, dyad))
    for file in chi_files: # for all kids file in all folder find those with rej3.set
        f = file.split('.')
        if f[0][-4:] == 'rej3':
            valid_chi_dynb_rej3.append(file.split('_')[1]) # store valid kid dyad
        elif f[0][-4:] == 'rej2':
            valid_chi_dynb_rej2.append(file.split('_')[1]) # store valid kid dyad
        else: 
            print('Dyad {} has no usable files (rej3 or rej2) for child'.format(dyad))
# Now that we know the child has the file, we go to the adult
for dyad in dyad_nb: # for all dyad folder in the main folder
    adu_files = os.listdir('{}{}_adult_FP'.format(data_path, dyad))
    for file in adu_files: # for all kids file in all folder find those with rej3.set
        f = file.split('.')
        if f[0][-4:] == 'rej3':
            valid_adu_dynb_rej3.append(file.split('_')[1]) # store valid adult dyad
        elif f[0][-4:] == 'rej2':
            valid_adu_dynb_rej2.append(file.split('_')[1]) # store valid kid dyad
        else: 
            print('Dyad {} has no usable files (rej3 or rej2) for adult'.format(dyad))

#%%
# Combine list of dyad with either rej3 or rej2 available
child_d = remove_double(valid_chi_dynb_rej2, valid_chi_dynb_rej3)
adult_d = remove_double(valid_adu_dynb_rej2, valid_adu_dynb_rej3)


#%%  Find the dyad that have a rej3.set file in both adult and child
dyad_to_study = set(child_d).intersection(adult_d)

#%%
to_process=[]
for dyad in dyad_to_study:
    files_chi = os.listdir('{}{}_child_FP/'.format(data_path, dyad))

    # for f in files_chi:
    #     if f.find("rej2.set")!=-1:
    #         tmp = f
    #     if f.find("rej3.set")!=-1:
    #         tmp = f

    for fname in files_chi:
        if fname.endswith('rej3.set'):
            tmp = fname
            break
    else:
        for fname in files_chi:
            if fname.endswith('rej2.set'):
                tmp = fname

    
    files_adu = os.listdir('{}{}_adult_FP/'.format(data_path, dyad))
    # for f in files_adu:
    #     if f.find("rej3.set")!=-1:
    #         to_process.append((dyad, tmp, f))

    for fname in files_adu:
        if fname.endswith('rej3.set'):
            to_process.append((dyad, tmp, fname))
            break
    else:
        for fname in files_adu:
            if fname.endswith('rej2.set'):
                to_process.append((dyad, tmp, fname))

np.savetxt("files_to_process.csv", 
           to_process,
           delimiter =",", 
           fmt ='% s')

# %%
ibc_metric = 'ccorr'
freq_bands = {'Theta': [4.0, 7.0],  # suggested by NB
              'Alpha': [7.0, 13.0], # suggested by NB
              'Beta': [13.5, 21.0],
              'Gamma': [30.0, 48.0]
              }

#%%

for dyad in to_process:
    print(dyad)
    epo_c = mne.read_epochs_eeglab('{}{}_child_FP/{}'.format(data_path, dyad[0], dyad[1]))
    epo_a = mne.read_epochs_eeglab('{}{}_adult_FP/{}'.format(data_path, dyad[0], dyad[2]))

    print("\nVerify equal epoch count: ")
    mne.epochs.equalize_epoch_counts([epo_c, epo_a])

    break

    print("\nVerify equal channel count: ")
    ch_to_drop_in_epo_c = list(set(epo_c.ch_names).difference(epo_a.ch_names))
    ch_to_drop_in_epo_a = list(set(epo_a.ch_names).difference(epo_c.ch_names))
    if len(ch_to_drop_in_epo_c) != 0:
        print('Dropping the following channel(s) in epo_c: {}'.format(ch_to_drop_in_epo_c))
        epo_c = epo_c.drop_channels(ch_to_drop_in_epo_c)
    else:
        print('No channel to drop in epo_c.')
    if len(ch_to_drop_in_epo_a) != 0:
        print('Dropping the following channel(s) in epo_a: {}'.format(ch_to_drop_in_epo_a))
        epo_a = epo_a.drop_channels(ch_to_drop_in_epo_a)
    else:
        print('No channel to drop in epo_a.')

    #  Initializing data and storage  #############################################################
    data_inter = np.array([])
    print(epo_c.get_data().shape, epo_a.get_data().shape)###########################
    data_inter = np.array([epo_c.get_data(), epo_a.get_data()]) #, dtype=mne.io.eeglab.eeglab.EpochsEEGLAB) # Deprecation warning

    # Computing analytic signal per frequency band ################################################
    print("- - - - > Computing analytic signal per frequency band ...")
    sampling_rate = epo_c.info['sfreq']
    np.save("{}/FINS-complex_signal/dyad_{}_complexsignal.npy".format(save_path, dyad[0]), complex_signal, allow_pickle=False)

    # Computing frequency- and time-frequency-domain connectivity ################################
    print("- - - - > Computing frequency- and time-frequency-domain connectivity ...")
    result = hypyp.analyses.compute_sync(complex_signal, mode=ibc_metric, epochs_average=True) # (n_freq, 2*n_channels, 2*n_channels)
    np.save("{}/FINS-IBC/dyad_{}_{}_IBC.npy".format(save_path, dyad[0], ibc_metric), result, allow_pickle=False)


# %%
ibc_results = []

for I, ibc_result_npy in enumerate(os.listdir(save_path+'/IBC/')):
    dy = ibc_result_npy.split('_')[1]
    ibc_results.append((dy, np.load(save_path+'/IBC/'+ibc_result_npy)))

#########################
# ibc_results is
#########################

# %%

print(ibc_results[0][1].mean(axis=(1,2)), ibc_results[1][1].mean(axis=(1,2)))
# %%

input('sure you wanna do that? (if not, you can still interrupt the process)')

all_psds = {
    'adult':[], 
    'child':[]
    }

# Create similar dict structure to store amount of epochs per subject
mean_epoch_count = deepcopy(all_psds)
dyad_order = []
desired_ch = ['Pz', 'P3', 'P4', 'P6', 'P7', 'P8'] # ['C3', 'C4', 'P3', 'P4', 'P7', 'P8', 'Cz', 'Pz', 'CPz']

for dyad in to_process:
    dyad_order.append(dyad[0])
    print("\n------", dyad[0])
    
    epo_c = mne.read_epochs_eeglab('{}{}_child_FP/{}'.format(data_path, dyad[0], dyad[1]))
    epo_c = epo_c.pick_channels(list(set(epo_c.ch_names).intersection(desired_ch)))
    psds, freq = mne.time_frequency.psd_welch(epo_c, fmin=4.0, fmax=50.0, n_fft=int(epo_c.info['sfreq']))
    mean_epoch_count['child'].append(psds.shape[0])
    all_psds['child'].append(psds)

    epo_a = mne.read_epochs_eeglab('{}{}_adult_FP/{}'.format(data_path, dyad[0], dyad[2]))
    epo_a = epo_a.pick_channels(list(set(epo_a.ch_names).intersection(desired_ch)))
    psds, freq = mne.time_frequency.psd_welch(epo_a, fmin=4.0, fmax=50.0, n_fft=int(epo_a.info['sfreq']))
    mean_epoch_count['adult'].append(psds.shape[0])
    all_psds['adult'].append(psds)
        
# %%
# %%##########
# Plot PSDs for the whole spectra
# ############

avr_psd = deepcopy(all_psds)

for role in ['child', 'adult']:
        print("Data in condition for {} is {} in average".format(role, np.round(np.mean(mean_epoch_count[role]), 3)))
        
        # average for each psd accross sensor / segments
        for i in range(len(avr_psd[role])):
            avr_psd[role][i] = np.mean(avr_psd[role][i], axis=(0, 1))

        tmp = pd.DataFrame(list(zip(freq, np.array(avr_psd[role]).sum(axis=0))), columns=['Freq','Power'])

        sns.lineplot(data=tmp, x="Freq", y="Power", label=' {}'.format(role)).set(title='PSD per condition & role')

# plt.legend(loc='upper right', title='Condition / Role') # Doesn't work

        # Data in condition for child is 206.3 in average
        # Data in condition for adult is 213.4 in average

del avr_psd
# %%
#%%
# Initialize fooof df with (roles, sensors, freqs)
roles = ['child', 'adult']

all_fooof_psds = np.zeros((len(roles), len(dyad_order), len(freq)))  # 'child' will be at index 0 and 'adult' at index 1


for role_idx, role in enumerate(roles):
    for i in range(len(all_psds[role])):

        all_fooof_psds[role_idx][i] = np.mean(all_psds[role][i], axis=(0, 1))



        # if all_psds[role][i].shape[1] != 22:
        #     pass
        # else:
        #     # Compute mean over the epoch dimension
        #     all_fooof_psds[role_idx][i] = np.mean(all_psds[role][i], axis=0)

#########   
# The shape of np.arr fooof_psds is now \
#               (2, 47) 
#               (role ['child', 'adult'], freqs)
#########
# %%
# %%##########
# Define NaNs policy, Initialize FOOOFGroup w/ desired settings, and filter psd df
# ############

def check_nans(data, nan_policy='zero'):
    """Check an array for nan values, and replace, based on policy."""

    # Find where there are nan values in the data
    nan_inds = np.where(np.isnan(data))

    # Apply desired nan policy to data
    if nan_policy == 'zero':
        data[nan_inds] = 0
    elif nan_policy == 'mean':
        data[nan_inds] = np.nanmean(data)
    else:
        raise ValueError('Nan policy not understood.')

    return data

fg = FOOOFGroup(
    peak_width_limits=[0.5, 10],
    min_peak_height=0.15,
    peak_threshold=1.,
    # max_n_peaks=20, 
    verbose=True)

# Define the frequency range to fit
freq_range = [np.min(freq), np.max(freq)]


# Fit the power spectrum model across all channels in condition of choice
looking_into_condi = 'adult'
if looking_into_condi == 'both conditions':
    fg.fit(freq, all_fooof_psds.mean(axis=0) , freq_range) # To fit all condi in SPEAKER
elif looking_into_condi == 'child':
    fg.fit(freq, all_fooof_psds[0] , freq_range) # To fit ES
elif looking_into_condi == 'adult':
    fg.fit(freq, all_fooof_psds[1] , freq_range) # To fit NS
else:
    print('Not understood.')

# Check the overall results of the group fits
fg.plot()
fg.print_results()
plt.savefig('{}FINS-Plots/FINS-FOOOF-plot/FINS_{}_FOOOFreport.pdf'.format(save_path, looking_into_condi))


# %%

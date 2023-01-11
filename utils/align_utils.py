import mne
import numpy as np
import pandas as pd
import tempfile

def EpochsEEGLAB_to_mneEpochsFIF (path):
    """
    Loads a SET file into a mne.io.eeglab.eeglab.EpochsEEGLAB object
    and converts it into a mne.Epochs instance.

    Arguments
    ----------
    path: str
        participant #1 fNIRS data path (directory)

    Returns
    --------
    mneEpochs:
        instance of mne.Epochs.
    """
    # read the file and get a mne.io.eeglab.eeglab.EpochsEEGLAB instance
    tmp = mne.io.read_epochs_eeglab(path)

    with tempfile.TemporaryDirectory() as tmpdir:
        # save it in FIF
        tmp.save(tmpdir+"tmp.fif", overwrite=True, verbose=None)
        
    # re-read it so it is now a mne.EpochsFIF
    return mne.read_epochs(tmpdir+"tmp.fif")

def create_initial_state_dict (last_state=list):
    state_dict = {}
    for i in last_state:
        state_dict[i] = i
    return state_dict

def take_a_step_back (state_dict=dict, rm_idx=list, is_first_round=bool) :
    
    # If this is the first round (i.e., the last rejection round)
    if is_first_round: 
        # Order the keys of the state_dict so we can iterate over them     
        existing_states = []
        for key in state_dict.keys():
            existing_states.append(key)
        existing_states.sort()

        # Add placeholders in the state_dict for the new state
        # added by introducing the removed states
        for new_key in range(len(rm_idx)):
            state_dict[existing_states[-1]+new_key+1] = existing_states[-1]+new_key+1

        # Order the keys of the state_dict so we can iterate over them 
        existing_states = []
        for key in state_dict.keys():
            existing_states.append(key)
        existing_states.sort()

        # Sort the states that were removed so we make sure we start by the 
        # lowest idx to reiterate shifting idx correctly
        rm_idx.sort()

        # For each index that was removed,
        for rmed in rm_idx:
            # Check all existing idx, and if the index
            # already exist, update it such that ...
            for existing in existing_states:
                # ... only the indexes that would be shifted by introducing 
                # a NaN sees their 'new' index substracted 1.
                # We substract 1 because in the persepctive of the initial df,
                # the index of a given state lost a rank because of removing a
                # state that was anterior to it.
                if existing > rmed:
                    if state_dict[existing] == 'NaN':
                        pass
                    else:
                        state_dict[existing] -= 1
            state_dict[rmed] = 'NaN'
    else:
        rm_idx.sort()

        for rmed in rm_idx:
            d2 = {key+1 if key >= rmed else key: value for key, value in state_dict.items()}
            d2[rmed] = 'NaN'
            state_dict = d2

    return state_dict

def revert_to_original_idx (last_state=list, removed_list=list, verbose=True):
    
    # First initialise the state dict containing the true idx as keys 
    # and their corresponding epoch (here the letters)
    state_dict = create_initial_state_dict(last_state)

    if verbose:
        print('Initial state:')
        for i in state_dict.keys():
            print('\t',i, state_dict[i])

    # revert "last_state" so we can beging by the last round
    removed_list.reverse()

    # Initialize the boolean "first_round_bool" to be true so the 
    # function knows it has to take the special step it needs to 
    # for 
    first_round_bool = True
    
    for rm_idx in removed_list:

        state_dict_arg = state_dict if first_round_bool else updated_state_dict

        updated_state_dict = take_a_step_back(
            state_dict=state_dict_arg, 
            rm_idx=rm_idx, 
            is_first_round=first_round_bool)
        
        if verbose:
            print('Updated state:')
            for i in updated_state_dict.keys():
                print('\t',i, updated_state_dict[i])
            
        # Not the first round anymore
        first_round_bool = False
    
    return updated_state_dict
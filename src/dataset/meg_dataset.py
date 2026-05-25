import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

_initialized_data = None

def pad_and_sort_batch(data_loader_batch, max_length, patches_number):
    """
    data_loader_batch should be a list of (sentences, meg, subject) tuples.
    Returns a padded tensor of sequences sorted from longest to shortest.
    """
    # we want max_length to be a multiple of patches_number
    max_length = int(np.ceil(max_length/patches_number)*patches_number)
    meg, sentences, subjects = zip(*data_loader_batch)

    meg_tensors = []
    for m in meg:
        padded_meg = []
        for seq in m:
            seq = torch.tensor(seq) 
            if seq.size(0) < max_length:
                pad_size = (0, max_length - seq.size(0))
                padded_seq = torch.nn.functional.pad(seq, pad_size)

            padded_meg.append(padded_seq)
        meg_tensors.append(torch.stack(padded_meg))

    meg_padded = pad_sequence(meg_tensors, batch_first=True, padding_value=0).permute(0, 1, 2)
    attention_mask = attention_mask_creation(meg_padded, patches_number)

    batches = len(meg_padded)
    channels = len(meg_padded[0])
    T = int(max_length/patches_number)

    meg_padded = torch.reshape(meg_padded, (batches, channels, patches_number, T))
    meg_padded = torch.swapaxes(meg_padded, 1,2)

    attention_mask = attention_mask.float()
    meg_padded = meg_padded.float()

    '''    
    meg.shape --> B, Ch, no Patches, T
    example --> 8, 208, 20, 438
    '''    
    
    return sentences, meg_padded, subjects, attention_mask

def attention_mask_creation(meg, patches_number):
    attention_mask = []
    max_length = len(meg[0][0])

    patch_size = int(max_length/patches_number)

    for i in range(meg.shape[0]):
        support_mask = []
        for patch_number in range(patches_number):

            lower_bound = patch_size*patch_number
            upper_bound = (patch_size*(patch_number+1)-1)
            actual_patch = meg[i,:,lower_bound:upper_bound]

            #check if all the meg values are 0
            all_zero = torch.all(actual_patch[0][0] == 0, dim=-1).all().item()

            if all_zero:
                for i in range(patches_number - patch_number):
                    support_mask.append(0)                
                break
            else:
                support_mask.append(1)
        attention_mask.append(support_mask)

    attention_mask = torch.tensor(attention_mask)
    return attention_mask

def pre_initialization(sentences, meg, subjects):
    global _initialized_data
    if _initialized_data is not None:
        return _initialized_data

    # WRAPPING INTO BLOCKS ----------------------------------------------------
    sentence_groups = {}
    for s, m, sub in zip(sentences, meg, subjects):
        if s not in sentence_groups:
            sentence_groups[s] = {"meg": [], "subjects": [], "sentences": []}
        sentence_groups[s]["meg"].append(m)
        sentence_groups[s]["subjects"].append(sub)
        sentence_groups[s]["sentences"].append(s)

    meg_blocks, sub_blocks, sentences_blocks = zip(*[
        (group["meg"], group["subjects"], group["sentences"])
        for group in sentence_groups.values()
    ])
    # -------------------------------------------------------------------------

    train_meg_blocks, test_meg_blocks, train_subjects_blocks, test_subjects_blocks, train_sentences_blocks, test_sentences_blocks = train_test_split(
        meg_blocks, sub_blocks, sentences_blocks, test_size=0.2, random_state=1)

    test_meg_blocks, val_meg_blocks, test_subjects_blocks, val_subjects_blocks, test_sentences_blocks, val_sentences_blocks = train_test_split(
        test_meg_blocks, test_subjects_blocks, test_sentences_blocks, test_size=0.5, random_state=1)

    # UNWRAPPING BLOCKS--------------------------------------------------------
    train_sentences = [sentence for block in train_sentences_blocks for sentence in block]
    train_meg = [meg for block in train_meg_blocks for meg in block]
    train_subjects = [subject for block in train_subjects_blocks for subject in block]    

    test_sentences = [sentence for block in test_sentences_blocks for sentence in block]
    test_meg = [meg for block in test_meg_blocks for meg in block]
    test_subjects = [subject for block in test_subjects_blocks for subject in block]

    val_sentences = [sentence for block in val_sentences_blocks for sentence in block]
    val_meg = [meg for block in val_meg_blocks for meg in block]
    val_subjects = [subject for block in val_subjects_blocks for subject in block]
    # -------------------------------------------------------------------------

    train_data = (train_sentences, train_meg, train_subjects)
    test_data = (test_sentences, test_meg, test_subjects)
    val_data = (val_sentences, val_meg, val_subjects)

    _initialized_data = (train_data, test_data, val_data)
    return _initialized_data

class meg_dataset(Dataset):
    def __init__(self, sentences, meg, subjects, phase):
        
        train_data, test_data, val_data = pre_initialization(sentences, meg, subjects)
        if phase == 'train':
            self.sentences, self.meg, self.subjects = train_data
        elif phase == 'test':
            self.sentences, self.meg, self.subjects = test_data
        elif phase == 'val':
            self.sentences, self.meg, self.subjects = val_data
        elif phase == 'all':
            self.sentences = sentences
            self.meg = meg 
            self.subjects = subjects 
        else:
            raise ValueError(f"Invalid phase: {phase}. Expected 'train', 'test', 'val', or 'all'.")

    def __getitem__(self, idx):
        return [self.meg[idx], self.sentences[idx], self.subjects[idx]]

    def __len__(self):
        return len(self.meg)
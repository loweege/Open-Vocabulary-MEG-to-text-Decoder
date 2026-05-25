import os
import numpy as np
from torch.utils.data import DataLoader

from data_utils.utils import unpickling
from dataset.meg_dataset import meg_dataset, pad_and_sort_batch


def cleansing(th, sentences, meg, subjects):
    # if a sentence has less or equal words than threshold, the data is cancelled
    filtered_data = [(s, m, sub) for s, m, sub in zip(sentences, meg, subjects) 
                     if len(s.split()) > th[0] and len(s.split()) < th[1] ]
    
    sentences[:], meg[:], subjects[:] = zip(*filtered_data) if filtered_data else ([], [], [])

def load_raw_data():
    path = 'bids_anonym/pickles'
    pickle_files = ["./"+root + "/"+ files[0] for root, dirs, files in os.walk(path) if len(files)>0]
    pickle_files.sort()
    pickle_files = [file for file in pickle_files if file != './bids_anonym/pickles/.DS_Store']

    sentences = [] 
    meg = []
    len_meg = [] 
    subjects = []

    for pickle_file in pickle_files[:1]:
        data = unpickling(pickle_file)
        for sentence, local_meg, subject in data:
            sentences.append(sentence)
            meg.append(local_meg)
            len_meg.append(local_meg.shape[1])
            subjects.append(subject)
    len_meg = np.array(len_meg)
    max_length = max(map(lambda x: len(x[2]), meg))

    th = [5, 500]
    cleansing(th, sentences, meg, subjects)
    return sentences, meg, subjects, max_length

def load_dataset(patches_number=20, 
                 batch_size=8,
                 n_workers=0):

    print("Reading dataset...")
    print(f"Number of pathces: {patches_number} - Batch size: {batch_size}")

    sentences, meg, subjects, max_length = load_raw_data()
    collate_fn = lambda batch: pad_and_sort_batch(batch, max_length, patches_number=patches_number)

    print(f"Max length: {max_length}")

    train_dataset = meg_dataset(sentences, meg, subjects, 'train')
    test_dataset = meg_dataset(sentences, meg, subjects, 'test')
    val_dataset = meg_dataset(sentences, meg, subjects, 'val')

    print(f"Train sentences: {train_dataset.__len__()}")
    print(f"Test sentences: {test_dataset.__len__()}")
    print(f"Val sentences: {val_dataset.__len__()}")


    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, 
                                  shuffle = True, collate_fn = collate_fn, num_workers=n_workers)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, 
                                 shuffle = False, collate_fn = collate_fn, num_workers=n_workers)
    val_dataloader = DataLoader(val_dataset, batch_size = 1, 
                                shuffle = False, collate_fn = collate_fn, num_workers=n_workers)
    
    return train_dataloader, val_dataloader, test_dataloader
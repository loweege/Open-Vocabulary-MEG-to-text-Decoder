import pandas as pd
import numpy as np
from .event import *
import pickle
import os
from dataclasses import asdict

def extract_sequence_info(events: pd.DataFrame, word: bool = True,
                          phoneme: bool = True) -> pd.DataFrame:
    """
    Extract information about word and/or phoneme sequences from an events DataFrame. 

    The following columns are created if they don't already exist:
        'word_index': index of a word in the sequence, e.g. in a sentence
        'word_sequence': actual sequence of words a word belongs to, e.g. a sentence
        'phoneme_id': index of a phoneme in a word

    The function returnes an updated EVENTS DATAFRAME
    """

    def is_missing(df, key):
        return key not in df.columns or all(df[key].isnull())

    events_out = events.copy()

    if word and (events.kind == 'word').any():
        missing_cols = [col for col in ['sequence_id', 'word'] if col not in events.columns]
        if missing_cols:
            raise ValueError(f'Columns \"{missing_cols}\" are required but were not found.')

        is_word = events.kind.isin(['word', 'multiplewords'])
        words = events.loc[is_word]

        if words.sequence_id.nunique() < 2:
            raise ValueError('Only one word sequence ID found.')

        for _, d in words.groupby("sequence_id"):
            # define word indices by making it compatible for multiple words
            if is_missing(d, "word_index"):  # Index of the word in the sequence
                indices = np.cumsum([0] + [len(w.split()) for w in d.word])
                events_out.loc[d.index, "word_index"] = indices[:-1]

            if is_missing(d, "word_sequence"):  # Sequence of words
                for uid in d.index:
                    events_out.loc[uid, "word_sequence"] = " ".join(d.word.values)

    if phoneme and (events.kind == 'phoneme').any():
        phonemes = events_out[events_out.kind == 'phoneme']
        if is_missing(phonemes, 'word_index'):
            raise ValueError('Column \"word_index\" is required but was not found.')

        for _, group in phonemes.groupby(['sequence_id', 'word_index']):
            if is_missing(group, 'phoneme_id'):
                events_out.loc[group.index, 'phoneme_id'] = range(len(group))

    return events_out

def _get_block_uid(events: pd.DataFrame) -> str:
    """
    Get block unique IDs for the events contained in a DataFrame.

    The unique ID of a block is either the concatenation of the words or filepaths it contains, or,
    if available and unique, the value in the 'sequence_uid' column.
    """
    if 'sequence_uid' in events.columns:  # Use existing sequence_uid, e.g. with Schoffelen2019
        unique_sequence_uids = events.sequence_uid.unique()
        if len(unique_sequence_uids) == 1:
            uid = unique_sequence_uids[0]
            return uid

    # Use concatenation of words or filepaths
    has_words = \
        events.condition.isin({'sentence', 'context', 'question', 'fixation', 'word_list'}) & (events.kind != 'phoneme')
    if not any(has_words):  # Use filepaths if there are no words in the block
        uid_ = [f for f in events.filepath.unique() if isinstance(f, str)]
        assert len(uid_), 'No filepath information available for defining block unique ID.'
        uid_ += [str(events.start.min())]
    else:
        uid_ = events.loc[has_words].word.astype(str)

    uid = ' '.join(uid_)

    return uid

def _create_blocks(events: pd.DataFrame, groupby: str) -> pd.DataFrame:
    """
    Create blocks from an EVENTS DATAFRAME.

    Blocks have a start, a duration, and a unique ID that can be used to identify its content.
    Blocks are used when splitting examples into training, validation and test sets to avoid
    creating segments that end in the middle of a sequence.

    Returns an updated EVENTS DATAFRAME that contains the created blocks.
    """

    # Find events that are valid block starts
    blocks = list()
    #for event in events.event.iter():
    for index, event in events.iterrows():
        if groupby == "sentence":
            block_start = (event.kind == "word") and (event.word_index == 0)
        elif groupby == "sound":
            block_start = event.kind == "sound"
        elif groupby == "fixation":
            block_start = event.condition == "fixation"
        elif groupby == 'sentence_or_sound':  # Used for Schoffelen2019
            block_start = (event.kind == 'sound') or (
                (event.kind == 'word') and (event.modality == 'visual') and
                (event.word_index == 0))
        else:
            block_start = False

        if block_start:
            blocks.append(event)

    eps = 1e-7
    event_stops = events.start + events.duration
    events_end = event_stops.max() + eps
    assert all(np.diff([b.start for b in blocks]) > 0), "events not sorted"
    block_stops = [b.start for b in blocks[1:]] + [events_end]

    # Add boundary unique ID
    block_events = list()
    for block, stop in zip(blocks, block_stops):
        # Create block unique ID based on all events contained in the block
        mask = (events.start >= block.start) & ((events.start + events.duration) < stop)
        uid = _get_block_uid(events[mask])
        block_info = asdict(  # Convert to Block object to apply checks
            Block(start=block.start, duration=stop - block.start, uid=uid,
                  language=block.language, modality=block.modality))
        block_events.append(block_info)

    blocks_df = pd.DataFrame(block_events)
    blocks_df['kind'] = 'block'
    blocks_df.duration.iat[-1] = float('inf')  # For compatibility with old API - last block has
    # infinite duration

    # Sort by start time
    events = pd.concat([events, blocks_df], axis=0, ignore_index=True)
    events.loc[events.kind == "block", "start"] -= eps  # Make sure blocks come before their events
    events = events.sort_values("start", ignore_index=True)
    events.loc[events.kind == "block", "start"] += eps  # Move back to real start time

    return events

def pickling(structure, output_dir, output_filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir,output_filename), 'wb') as handle:
        pickle.dump(structure, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('write to:', os.path.join(output_dir,output_filename))

def unpickling(pickle_file):
        '''
        read the pickle file in input an return a dataframe version of it
        '''
        with open(pickle_file, 'rb') as handle:
            data = pickle.load(handle)
        #data = pd.DataFrame(data)
        return data
import mne_bids
import pandas as pd
from pathlib import Path
from scipy import signal
from data_utils.utils import _create_blocks, extract_sequence_info, pickling

'''
INSTRUCTIONS:

1) Ensure the dataset is located within your project folder.
2) The dataset must be named 'bids_anonym'
3) data_path? --> leave it blank
'''

class PATHS:
    path_file = Path("./data_path.txt")
    if not path_file.exists():
        data = Path(input("data_path?"))
        assert data.exists()
        with open(path_file, "w") as f:
            f.write(str(data) + "\n")
    with open(path_file, "r") as f:
        data = Path(f.readlines()[0].strip("\n"))

    assert data.exists()

    bids = data / "bids_anonym"


ph_info = pd.read_csv("phoneme_info.csv")
subjects = pd.read_csv(PATHS.bids / "participants.tsv", sep="\t")
subjects = subjects.participant_id.apply(lambda x: x.split("-")[1]).values


def _get_words(subject, session, task):
    bids_path = mne_bids.BIDSPath(
                subject=subject,
                session=str(session),
                task=str(task),
                datatype="meg",
                root=PATHS.bids,
            )
    try:
        raw = mne_bids.read_raw_bids(bids_path)
    except FileNotFoundError:
        print("missing", subject, session, task)

    raw = raw.pick_types(
                meg=True, misc=False, eeg=False, eog=False, ecg=False
            )

    raw.load_data().filter(0.5, 30.0, n_jobs=1)

    events = list()
    for annot in raw.annotations:
        event = eval(annot.pop("description"))
        event['start'] = annot['onset']
        event['duration'] = annot['duration']
        if event["kind"] == "sound":
            stem, _, ext = event["sound"].lower().rsplit(".", 2)
        events.append(event)
    events_df = pd.DataFrame(events)
    events_df[['language', 'modality']] = 'english', 'audio' #shape -> shape + 2
    events_df = extract_sequence_info(events_df) #shape -> shape + 2 (words_sequence, phoneme_id)
    events_df = _create_blocks(events_df, groupby='sentence') #shape -> shape + 1 (uid)
    
    sentences = events_df[events_df['kind'] == 'block']

    meta = list()
    for annot in raw.annotations:
        d = eval(annot.pop("description"))
        for k, v in annot.items():
            assert k not in d.keys()
            d[k] = v
        meta.append(d)
    meta = pd.DataFrame(meta)
    meta["intercept"] = 1.0

    # compute voicing
    phonemes = meta.query('kind=="phoneme"')
    assert len(phonemes)
    for ph, d in phonemes.groupby("phoneme"):
        ph = ph.split("_")[0]
        match = ph_info.query("phoneme==@ph")
        assert len(match) == 1
        meta.loc[d.index, "voiced"] = match.iloc[0].phonation == "v"

    # compute word frquency and merge w/ phoneme
    meta["is_word"] = False
    words = meta.query('kind=="word"').copy()

    return raw, sentences, words



#----------------------------herethemain--------------------------------------------------
if __name__ == "__main__":

    #only 2 subjects for testing purposes
    temporal_subjects = subjects[19:23] 

    for subject in temporal_subjects:
        structure = []
        for session in range(2):
            # to handle subjects who have participated at only one recording session
            if ((subject == '03' or  subject == '12' or subject == '16' 
                 or subject == '20' or subject == '21' or subject == '22') 
                 and session == 1):
                continue

            for story in range(4):

                print('-----subject-----')
                print(subject)
                print('-----session-----')
                print(session)
                print('------story------')
                print(story)  

                raw, sentences, words = _get_words(subject, session, story)

                uid = sentences.uid
                raw_data = raw.get_data()

                # Resample MEG data from 1000 Hz to 120 Hz
                fs_new = 120 
                num_samples_new = int(raw_data.shape[1] * fs_new / 1000) 
                raw_data = signal.resample(raw_data, num_samples_new, axis=1)

                #------------------------------------------------------------------------------------------------
                #meg retrieval for each sentence.
                for i in range(len(sentences)):
                    start_time = round(sentences.iloc[i]['start'] * fs_new)

                    #if this is not the last sentence
                    if (i != len(sentences) - 1):
                        end_time = round(start_time + (sentences.iloc[i]['duration'] *  fs_new))
                    else:    
                        end_time = round((words.iloc[-1]['onset'] + words.iloc[-1]['duration']) * fs_new)

                    sentence = uid.iloc[i]
                    local_meg = raw_data[:,start_time : end_time]

                    structure.append([sentence, local_meg, subject])
                    

        #Do pickling only if data has not been stored yet
        pickling_already_done = True

        if not pickling_already_done:
            output_dir = f'./bids_anonym/pickles/{subject}'
            output_filename = f'sub{subject}_raw_meg.pickle'
            pickling(structure, output_dir, output_filename)
            structure = None


# Open-Vocabulary-MEG-to-text-Decoder

Our brain constantly produces signals when we speak, listen to music, watch a
TV show, and even when we sleep. These signals come from the electromagnetic
spectrum. In particular, one type of signal comes from magnetoencephalography
(MEG). Decoding text or speech from brain activity and creating a brain-to-text
model has been the focus of several research efforts over the years. Although
diverse relevant studies already exist, demonstrating the capabilities of models
exploiting invasive and semi-invasive retrieval techniques, our focus in this
work is on non-invasive techniques. We aim to build a model capable
of predicting text from brain signals. Our model is an open vocabulary
magnetoencephalography-to-text decoding model. We utilized MEG signals from
the MEG-MASC dataset, one of the few datasets containing high-quality signals
thathavebeenusedinawidevarietyofstudies. Weconsideredonlytherecordings
of the first 23 subjects out of the 30 contained in the dataset. Each subject was
engaged in listening tasks. Before the experiment, all participants listened to 20
seconds of each of the speaking voices to familiarize themselves with the voices.
To ensure that the participants were attentive to the stories, they answered a
two-alternativeforced-choicequestionrelatedtothestorycontentevery3minutes
by pressing a button. Our approach involves a Sequence-to-Sequence model
composed of both an Encoder, which is a Transformer-based Encoder supported
by a Convolutional Block, and a Decoder, which is a T5 Transformer-based
Decoder. Our model achieves a ROUGE-1-R score of 31.08%, a ROUGE-1-P of
9.392%, and a ROUGE-1-F of 13.52%.

## Architecture
<img width="807" height="479" alt="Screenshot 2026-05-25 at 14 42 27" src="https://github.com/user-attachments/assets/9d07cf61-d67c-4462-a9f5-4b5b92d7c703" />



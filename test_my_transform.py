import torch
import torchaudio
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from lab4_proto import strToInt


# spectrograms is a tensor of shape B x C x M x T where B is batch, C is channels (=1),
# T is time (frames) and M is mel bands (=80).

# labels is a tensor of shape B x L where L is label length.

# input_lengths list of integers Li = Ti /2 where Ti is the number of frames in spectrogram
# i (before padding)

# label_lengths list of integers corresponding to the lengths of the label strings.

def dataProcessing(data, transform):
    '''
    process a batch of speech data
    arguments:
        data: list of tuples, representing one batch. Each tuple is of the form
            (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
        transform: audio transform to apply to the waveform
    returns:
        a tuple of (spectrograms, labels, input_lengths, label_lengths)
        -   spectrograms - tensor of shape B x C x T x M
            where B=batch_size, C=channel, T=time_frames, M=mel_band.
            spectrograms are padded the longest length in the batch.
        -   labels - tensor of shape B x L where L is label_length.
            labels are padded to the longest length in the batch.
        -   input_lengths - list of half spectrogram lengths before padding
        -   label_lengths - list of label lengths before padding
    '''

    # initialize the lists
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for i in range(len(data)):
        # get the waveform and sample rate
        waveform, sample_rate = data[i][0], data[i][1]
        # apply the transform to the waveform
        spectrogram = transform(waveform)

        spectrogram = spectrogram.squeeze(0).transpose(0, 1)

        # get the utterance and speaker id
        utterance, speaker_id = data[i][2], data[i][3]
        # get the chapter id and utterance id
        chapter_id, utterance_id = data[i][4], data[i][5]
        # append the spectrogram to the list
        spectrograms.append(spectrogram)
        # append the labels to the list
        labels.append(strToInt(utterance))
        # append the input lengths to the list
        input_lengths.append(spectrogram.shape[0] // 2)
        # append the label lengths to the list
        label_lengths.append(len(labels[-1]))

    # pad the spectrograms to the longest length in the batch
    spectrograms = pad_sequence(spectrograms, batch_first=True)
    # pad the labels to the longest length in the batch
    labels = pad_sequence([torch.tensor(l) for l in labels], batch_first=True)
    # convert the spectrograms to a tensor
    spectrograms = spectrograms.unsqueeze(1).transpose(2, 3)
    # convert the input lengths to a tensor
    input_lengths = torch.tensor(input_lengths)
    # convert the label lengths to a tensor
    label_lengths = torch.tensor(label_lengths)
    # return the spectrograms, labels, input lengths, and label lengths
    return spectrograms, labels, input_lengths, label_lengths


example = torch.load('lab4_example.pt')

mel_spec_transform = torchaudio.transforms.MelSpectrogram(16000, n_mels=80)

transform_augment = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(16000, n_mels=80),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35),
)

ret = dataProcessing(example["data"], mel_spec_transform)



import torch
import torchaudio
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# DT2119, Lab 4 End-to-end Speech Recognition

# Variables to be defined --------------------------------------
''' 
train-time audio transform object, that transforms waveform -> spectrogram, with augmentation
''' 
train_audio_transform = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(16000, n_mels=80),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35),
)
'''
test-time audio transform object, that transforms waveform -> spectrogram, without augmentation 
'''
test_audio_transform = torchaudio.transforms.MelSpectrogram(16000, n_mels=80)

# Functions to be implemented ----------------------------------

def intToStr(labels):
    '''
        convert list of integers to string
    Args: 
        labels: list of ints
    Returns:
        string with space-separated characters
    '''
    ret = []
    for i in labels:
        if i < 0 or i > 27:
            raise ValueError("Invalid character in string: {}".format(i))
        if i == 0:
            r = "'"
        elif i == 1:
            r = "_"
        else:
            r = chr(i + 95)
        ret.append(r)
    return ret


def strToInt(text):
    '''
        convert string to list of integers
    Args:
        text: string
    Returns:
        list of ints
    '''
    ret = []

    for i in text:
        if i == "'":
            r = 0
        elif i == "_" or i == " ":
            r = 1
        else:
            r = ord(i.lower()) - 95

        if r < 0 or r > 27:
            raise ValueError("Invalid character in string: {}".format(i))
        ret.append(int(r))

    return ret


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

    
def greedyDecoder(output, blank_label=28):
    '''
    decode a batch of utterances 
    arguments:
        output: network output tensor, shape B x T x C where B=batch_size, T=time_steps, C=characters
        blank_label: id of the blank label token
    returns:
        list of decoded strings
    '''
    preds = output.argmax(dim=-1)

    decoded_bath = []
    for batch in preds:
        prev = blank_label
        decoded = []
        for p in batch:
            if p != prev and p != blank_label:
                decoded.append(p)
            prev = p
        decoded_bath.append(intToStr(decoded))
    return decoded_bath


def levenshteinDistance(ref, hyp):
    """
    calculate levenshtein distance (edit distance) between two sequences
    arguments:
        ref: reference sequence
        hyp: sequence to compare against the reference
    output:
        edit distance (int)
    """
    # Initialize a matrix matrix with dimensions (len(sequence1) + 1) × (len(sequence2) + 1)
    mat = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)

    # Fill the first row of matrix with values ranging from 0 to len(sequence2)
    mat[0, :] = np.arange(len(hyp) + 1)

    # Fill the first column of matrix with values ranging from 0 to len(sequence1)
    mat[:, 0] = np.arange(len(ref) + 1)

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            # If the characters are the same, no cost is incurred
            if ref[i - 1] == hyp[j - 1]:
                mat[i, j] = mat[i - 1, j - 1]
            else:
                # If the characters are different, calculate the cost of insertion, deletion, and substitution
                insertion = mat[i, j - 1] + 1
                deletion = mat[i - 1, j] + 1
                substitution = mat[i - 1, j - 1] + 1

                # Take the minimum of the three costs
                mat[i, j] = min(insertion, deletion, substitution)
    # The edit distance is the value in the bottom-right corner of the matrix
    return mat[len(ref), len(hyp)]

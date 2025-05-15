import torchaudio
train_dataset = torchaudio.datasets.LIBRISPEECH('.',
url='train-clean-100', download=True)
val_dataset = torchaudio.datasets.LIBRISPEECH('.',
url='dev-clean', download=True)
test_dataset = torchaudio.datasets.LIBRISPEECH('.',
url='test-clean', download=True)
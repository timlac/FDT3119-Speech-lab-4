import torchaudio

train_dataset = torchaudio.datasets.LIBRISPEECH(".", url='train-clean-100', download=True)

example = train_dataset[0]
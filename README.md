# at16k
Pronounced as ***at sixteen k***

# What is at16k?
at16k is a Python library to perform automatic speech recognition or speech to text conversion. The goal of this project is to provide develoeprs with a production quality speech-to-text library.

# Installation
It is recommended that you install at16k in a virtual environment.

## Prerequisites
- Python = 3.6 (not tested on other versions)
- Tensorflow = 1.14
- Scipy (for reading wav files)

## Install via pip
```
$ pip install at16k
```

## Install from source
Requires: [poetry](https://github.com/sdispater/poetry)
```
$ git clone https://github.com/at16k/at16k.git
$ poetry env use python3.6
$ poetry install
```

# Download models
Currently, two models are available for speech to text conversion.
- en_8k (Trained on english audio recorded at 8 KHz)
- en_16k (Trained on english audio recorded at 16 KHz)

To download all the models:
```
$ python -m at16k.download all
```
Alternatively, you can download only the model you need. For example:
```
$ python -m at16k.download en_8k
$ python -m at16k.download en_16k
```

# Preprocessing audio files
at16k accepts wav files with the following spces:
- Channels: 1
- Bits per sample: 16
- Sample rate: 8000 (en_8k) or 16000 (en_16k)

Use ffmpeg to convert your audio/video files to an acceptable format. For example,
```
# For 8 KHz
$ ffmpeg -i <input_file> -ar 8000 -ac 1 -ab 16 <output_file>

# For 16 KHz
$ ffmpeg -i <input_file> -ar 16000 -ac 1 -ab 16 <output_file>
```

# Usage

## Command line

## Library API
```
from at16k.api import SpeechToText

# One-time initialization
STT = SpeechToText('en_16k') # or en_8k

# Run STT on an audio file, returns a dict
print(STT('./samples/test.wav'))
```
Check [example.py](example.py) for details on how to use the API.
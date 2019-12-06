# at16k
Pronounced as ***at sixteen k***

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/GlibAI/at16k/graphs/commit-activity)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyPI license](https://img.shields.io/pypi/l/at16k.svg)](https://pypi.python.org/pypi/at16k/)
[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
<img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat">
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/at16k.svg)
[![Downloads](https://pepy.tech/badge/at16k)](https://pepy.tech/project/at16k)

# What is at16k?
at16k is a Python library to perform automatic speech recognition or speech to text conversion. The goal of this project is to provide the community with a production quality speech-to-text library.

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
There are two ways to invoke at16k speech-to-text via the command line.
```
at16k-convert -i <input_wav_file> -m <model_name>
```
Alternatively,
```
python -m at16k.bin.speech_to_text -i <input_wav_file> -m <model_name>
```
## Library API
```
from at16k.api import SpeechToText

# One-time initialization
STT = SpeechToText('en_16k') # or en_8k

# Run STT on an audio file, returns a dict
print(STT('./samples/test_16k.wav'))
```
Check [example.py](https://github.com/at16k/at16k/blob/master/example.py) for details on how to use the API.

# Limitations

The max duration of your audio file should be less than **30 seconds** when using **en_8k**, and less than **15 seconds** when using **en_16k**. An error will not be thrown if the duration exceeds the limits, however, your transcript may contain errors and missing text.

# License

This software is distributed under the MIT license.

# Acknowledgements

We would like to thank [Google TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc) program for providing access to cloud TPUs.

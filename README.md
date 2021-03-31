[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/GlibAI/at16k/graphs/commit-activity)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyPI license](https://img.shields.io/pypi/l/at16k.svg)](https://pypi.python.org/pypi/at16k/)
[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
<img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat">
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/at16k.svg)
[![Downloads](https://pepy.tech/badge/at16k)](https://pepy.tech/project/at16k)

# at16k
Pronounced as ***at sixteen k***.

# What is at16k?
at16k is a Python library to perform automatic speech recognition or speech to text conversion. The goal of this project is to provide the community with a production quality speech-to-text library.

# Installation
It is recommended that you install at16k in a virtual environment.

## Prerequisites
- Python >= 3.6
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
Currently, three models are available for speech to text conversion.
- en_8k (Trained on English audio recorded at 8 KHz, supports offline ASR)
- en_16k (Trained on English audio recorded at 16 KHz, supports offline ASR)
- en_16k_rnnt (Trained on English audio recorded at 16 KHz, supports real-time ASR)

To download all the models:
```
$ python -m at16k.download all
```
Alternatively, you can download only the model you need. For example:
```
$ python -m at16k.download en_8k
$ python -m at16k.download en_16k
$ python -m at16k.download en_16k_rnnt
```
By default, the models will be downloaded and stored at <HOME_DIR>/.at16k. To override the default, set the environment variable AT16K_RESOURCES_DIR.
For example:
```
$ export AT16K_RESOURCES_DIR=/path/to/my/directory
```
You will need to reuse this environment variable while using the API via command-line, library or REST API.

# Preprocessing audio files
at16k accepts wav files with the following specs:
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
at16k supports two modes for performing ASR - offline and real-time. And, it comes with a handy command line utility to quickly try out different models and use cases.

Here are a few examples -
```
# Offline ASR, 8 KHz sampling rate
$ at16k-convert -i <path_to_wav_file> -m en_8k

# Offline ASR, 16 KHz sampling rate
$ at16k-convert -i <path_to_wav_file> -m en_16k

# Real-time ASR, 16 KHz sampling rate, from a file, beam decoding
$ at16k-convert -i <path_to_wav_file> -m en_16k_rnnt -d beam

# Real-time ASR, 16 KHz sampling rate, from mic input, greedy decoding (requires pyaudio)
$ at16k-convert -m en_16k_rnnt -d greedy
```
If the ***at16k-convert*** binary is not available for some reason, replace it with - 
```
python -m at16k.bin.speech_to_text ...
```

## Library API
Check [this file](https://github.com/at16k/at16k/blob/master/at16k/bin/speech_to_text.py) for examples on how to use at16k as a library.

# Limitations

The max duration of your audio file should be less than **30 seconds** when using **en_8k**, and less than **15 seconds** when using **en_16k**. An error will not be thrown if the duration exceeds the limits, however, your transcript may contain errors and missing text.

# License

This software is distributed under the MIT license.

# Acknowledgements

We would like to thank [Google TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc) program for providing access to cloud TPUs.

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

# Usage

## Command line

## Library API
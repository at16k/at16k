"""
Audio file handler
"""

import scipy.io.wavfile as wavfile
import numpy as np


class Media:
    """
    Media: I/O functionality to read/write audio files
    """

    def __init__(self, file_path, dtype=None):
        self.file_path = file_path
        self._dtype = dtype

    @property
    def sample_rate(self):
        """
        Sample rate of audio file
        """
        sample_rate, _ = wavfile.read(self.file_path)
        return sample_rate

    @property
    def waveform(self):
        """
        Raw waveform of entire audio file
        """
        _, waveform = wavfile.read(self.file_path)
        shape = np.shape(waveform)
        if len(shape) > 1:
            waveform = np.mean(waveform, axis=1)
        if self._dtype is None:
            return waveform
        if waveform.dtype not in [self._dtype]:
            waveform = waveform.astype(self._dtype) / np.iinfo(waveform.dtype).max
        return waveform

    @property
    def duration(self):
        """
        Length of the audio file (in seconds)
        """
        return len(self.waveform) / float(self.sample_rate)

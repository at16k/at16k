"""
Audio segment
"""


class Segment():
    """
    Audio segment
    """

    def __init__(self, waveform, boundaries, sample_rate, channel):
        self._waveform = waveform
        self._boundaries = boundaries
        self._sample_rate = sample_rate
        self._channel = channel

    @property
    def waveform(self):
        """
        Segment waveform
        """
        return self._waveform

    @property
    def boundaries(self):
        """
        Segment begin and end timestamp (in seconds)
        """
        return self._boundaries

    @property
    def sample_rate(self):
        """
        Sample rate
        """
        return self._sample_rate

    @property
    def channel(self):
        """
        Channel number (starts from 0)
        """
        return self._channel

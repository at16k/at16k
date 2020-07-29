"""
Speech-to-text pipeline
"""
import numpy as np
from at16k.core.media import Media
from at16k.core.live_model import LiveModel


class LiveSpeechToText:
    """
    Live speech-to-text
    """

    def __init__(self, model_name, buffer_size=4096, filter_non_speech=False, faster=True, warm_up=True):
        self._buffer_size = buffer_size
        self._model = self._load_model(model_name, filter_non_speech, faster)
        if warm_up:
            self._do_warmup()

    @staticmethod
    def _load_model(model_name, filter_non_speech, faster):
        model = LiveModel(model_name, filter_non_speech=filter_non_speech, faster=faster)
        return model

    def _do_warmup(self):
        samples = [0. for _ in range(4096)]
        self._model(samples, None)

    def from_file(self, file_path):
        """
        Live transcribe from file
        """
        model = self._model
        media = Media(file_path, dtype=np.float32)
        samples = media.waveform.ravel().tolist()
        buffer_size = self._buffer_size
        context = None
        while samples:
            text, context = model(samples[:buffer_size], context)
            samples = samples[buffer_size:]
            yield {"text": text}

    def from_buffer(self, buffer, context, dtype='<i2', is_buffer=True):
        """
        Transcribe from buffer
        """
        model = self._model
        if is_buffer:
            samples = np.frombuffer(buffer, dtype=dtype).reshape(-1, 1)
            if samples.dtype not in [np.float32]:
                samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
            samples = samples.ravel().tolist()
        else:
            samples = buffer
        text, context = model(samples, context)
        return text, context

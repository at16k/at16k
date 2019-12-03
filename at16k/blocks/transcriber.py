"""
Transcriber
"""

import tensorflow as tf
from at16k.core.segment import Segment
from at16k.core.model import Model

class Transcriber():
    """
    Transcribe an audio segment

    Returns:
        [type] -- [description]
    """

    def __init__(self, model: Model):
        self._model = model

    @staticmethod
    def _make_example(waveform):
        features = {
            "waveforms": tf.train.Feature(float_list=tf.train.FloatList(value=waveform))
        }
        example = tf.train.Example(
            features=tf.train.Features(feature=features))
        return example.SerializeToString()

    @staticmethod
    def _decode_example(example, model):
        pred_fn = model.pred_fn
        output_fn = model.output_fn
        predictions = pred_fn({'input': [example]})
        output = list(predictions['outputs'])[0]
        score = list(predictions['scores'])[0]
        text = output_fn(output, strip_extraneous=True)
        return text, float(score)

    def __call__(self, segment: Segment):
        model = self._model
        waveform = segment.waveform
        example = self._make_example(waveform)
        text, score = self._decode_example(example, model)
        return text, score

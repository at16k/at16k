"""
Load models for ASR
"""

import os
from pathlib import Path

import tensorflow as tf
from at16k.utils.text_encoder import SubwordTextEncoder


class Model():
    """ASR Model
    Two propoerties/methods are exposed:
    1) pred_fn: converts a waveform to a list of ids
    2) output_fn: converts list of ids to text
    """

    def __init__(self, name):
        self.name = name
        self._pred_fn = self._build_pred_fn()
        self._output_fn = self._build_output_fn()

    @property
    def pred_fn(self):
        """
        Converts waveform to list of ids
        """
        return self._pred_fn

    @property
    def output_fn(self):
        """
        Converts list of ids to text
        """
        return self._output_fn

    @staticmethod
    def _get_model_dir(name):
        if 'AT16K_RESOURCES_DIR' in os.environ:
            base_dir = os.environ['AT16K_RESOURCES_DIR']
        else:
            sys_home_dir = str(Path.home())
            base_dir = os.path.join(sys_home_dir, '.at16k')
        model_dir = os.path.join(base_dir, name)
        assert os.path.exists(model_dir), ('%s model does not exist at %s' % (name, model_dir))
        return model_dir

    def _build_pred_fn(self):
        name = self.name
        model_dir = self._get_model_dir(name)
        model_path = os.path.join(model_dir, 'export')
        sub_dirs = os.listdir(model_path)
        sub_dirs.sort()
        latest_model_path = os.path.join(model_path, sub_dirs[-1])
        pred_fn = tf.contrib.predictor.from_saved_model(latest_model_path)
        return pred_fn

    def _build_output_fn(self):
        name = self.name
        model_dir = self._get_model_dir(name)
        model_path = os.path.join(model_dir, 'vocab', 'bpe.1000.t2t')
        vocab_model = SubwordTextEncoder(filename=model_path)
        return vocab_model.decode

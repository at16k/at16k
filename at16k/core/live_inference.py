import os
import numpy as np
import tensorflow as tf


class LiveInferrer:
    def __init__(self, params, model_dir):
        self._params = params
        self._sessions, self._nodes = self._make_sessions(model_dir)
        self._model_dir = model_dir

    @staticmethod
    def _load_graph(frozen_graph_filename):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="prefix")
        return graph

    def _make_audio_features_session(self, graph_path):
        _nodes = {}
        _graph = self._load_graph(graph_path)
        _nodes['inputs'] = _graph.get_tensor_by_name('prefix/samples:0')
        _nodes['outputs'] = _graph.get_tensor_by_name('prefix/feats:0')
        _session = tf.Session(graph=_graph)
        return _session, _nodes

    def _make_delta_features_session(self, graph_path):
        _nodes = {}
        _graph = self._load_graph(graph_path)
        _nodes['inputs'] = _graph.get_tensor_by_name('prefix/raw_feats:0')
        _nodes['outputs'] = _graph.get_tensor_by_name('prefix/delta_feats:0')
        _session = tf.Session(graph=_graph)
        return _session, _nodes

    def _make_audio_encoder_session(self, graph_path):
        _nodes = {}
        _graph = self._load_graph(graph_path)
        _nodes['inputs'] = _graph.get_tensor_by_name(
            'prefix/audio_encoder/a_inputs:0')
        _nodes['inputs_states'] = _graph.get_tensor_by_name(
            'prefix/audio_encoder/a_inputs_states:0')
        _nodes['outputs'] = _graph.get_tensor_by_name(
            'prefix/audio_encoder/a_outputs:0')
        _nodes['outputs_states'] = _graph.get_tensor_by_name(
            'prefix/audio_encoder/a_states:0')
        _session = tf.Session(graph=_graph)
        return _session, _nodes

    def _make_text_encoder_session(self, graph_path):
        _nodes = {}
        _graph = self._load_graph(graph_path)
        _nodes['inputs'] = _graph.get_tensor_by_name(
            'prefix/text_encoder/t_inputs:0')
        _nodes['inputs_lengths'] = _graph.get_tensor_by_name(
            'prefix/text_encoder/t_inputs_lengths:0')
        _nodes['inputs_states'] = _graph.get_tensor_by_name(
            'prefix/text_encoder/t_inputs_states:0')
        _nodes['outputs'] = _graph.get_tensor_by_name(
            'prefix/text_encoder/t_outputs:0')
        _nodes['outputs_states'] = _graph.get_tensor_by_name(
            'prefix/text_encoder/t_states:0')
        _session = tf.Session(graph=_graph)
        return _session, _nodes

    def _make_joint_encoder_session(self, graph_path):
        _nodes = {}
        _graph = self._load_graph(graph_path)
        _nodes['a_inputs'] = _graph.get_tensor_by_name(
            'prefix/rnnt_logits/a_logits:0')
        _nodes['t_inputs'] = _graph.get_tensor_by_name(
            'prefix/rnnt_logits/t_logits:0')
        _nodes['logits'] = _graph.get_tensor_by_name('prefix/joint_logits:0')
        _nodes['probs'] = _graph.get_tensor_by_name('prefix/joint_log_probs:0')
        _session = tf.Session(graph=_graph)
        return _session, _nodes

    def _make_sessions(self, model_dir):
        _f_session, _f_nodes = self._make_audio_features_session(
            os.path.join(model_dir, 'audio_features.graph.pb'))
        _d_session, _d_nodes = self._make_delta_features_session(
            os.path.join(model_dir, 'delta_features.graph.pb'))
        _a_session, _a_nodes = self._make_audio_encoder_session(
            os.path.join(model_dir, 'audio_encoder.graph.pb'))
        _t_session, _t_nodes = self._make_text_encoder_session(
            os.path.join(model_dir, 'text_encoder.graph.pb'))
        _j_session, _j_nodes = self._make_joint_encoder_session(
            os.path.join(model_dir, 'joint_encoder.graph.pb'))
        _sessions = {
            'f': _f_session,
            'd': _d_session,
            'a': _a_session,
            't': _t_session,
            'j': _j_session
        }
        _nodes = {
            'f': _f_nodes,
            'd': _d_nodes,
            'a': _a_nodes,
            't': _t_nodes,
            'j': _j_nodes
        }
        return _sessions, _nodes

    def infer_features(self, inputs):
        _session = self._sessions['f']
        _nodes = self._nodes['f']
        outputs = _session.run(_nodes['outputs'], feed_dict={
            _nodes['inputs']: inputs
        })
        return outputs

    def infer_delta_features(self, inputs):
        _session = self._sessions['d']
        _nodes = self._nodes['d']
        outputs = _session.run(_nodes['outputs'], feed_dict={
            _nodes['inputs']: inputs
        })
        return outputs

    def infer_audio_encoder(self, inputs, inputs_states=None):
        _session = self._sessions['a']
        _nodes = self._nodes['a']
        _params = self._params
        if inputs_states is None:
            inputs_states = np.zeros((_params['audio_encoder_layers'], 2, 1, _params['audio_encoder_units']))
        outputs, outputs_states = _session.run([_nodes['outputs'], _nodes['outputs_states']], feed_dict={
            _nodes['inputs']: inputs,
            _nodes['inputs_states']: inputs_states
        })
        return outputs, outputs_states

    def infer_text_encoder(self, inputs, inputs_lengths, inputs_states=None):
        _session = self._sessions['t']
        _nodes = self._nodes['t']
        _params = self._params
        if inputs_states is None:
            inputs_states = np.zeros((_params['text_encoder_layers'], 2, 1, _params['text_encoder_units']))
        outputs, outputs_states = _session.run([_nodes['outputs'], _nodes['outputs_states']], feed_dict={
            _nodes['inputs']: inputs,
            _nodes['inputs_lengths']: inputs_lengths,
            _nodes['inputs_states']: inputs_states
        })
        return outputs, outputs_states

    def infer_joint_encoder(self, a_inputs, t_inputs):
        _session = self._sessions['j']
        _nodes = self._nodes['j']
        logits, probs = _session.run([_nodes['logits'], _nodes['probs']], feed_dict={
            _nodes['a_inputs']: a_inputs,
            _nodes['t_inputs']: t_inputs
        })
        return logits, probs

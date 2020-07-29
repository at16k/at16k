"""
Load models for ASR
"""

import time
import json
import os
from pathlib import Path
import numpy as np
import sentencepiece as spm
from at16k.core.live_inference import LiveInferrer


class BeamCandidate:
    """
    Single Beam Candidate
    """

    def __init__(self, seq=None, hidden=None, null_id=0):
        if seq is None:
            self.text_outs = []
            self.preds = [int(null_id)]
            self.text_state = hidden
            self.log_prob = 0.
        else:
            self.text_outs = seq.text_outs[:]
            self.preds = seq.preds[:]
            self.text_state = seq.text_state
            self.log_prob = seq.log_prob


class LiveModel:
    """
    Live ASR Model (real-time)
    """

    def __init__(self, name, filter_non_speech=True, faster=True, beams=10):
        _model_dir = self._get_model_dir(name)
        _params = self._load_hparams(_model_dir)
        _vocab = self._load_vocab(_model_dir)
        self.name = name
        self._faster = faster
        self._num_beams = beams
        self._filter_non_speech = filter_non_speech
        self._inferrer = LiveInferrer(_params, _model_dir)
        self._params = _params
        self._vocab = _vocab

    @staticmethod
    def _get_model_dir(name):
        if 'AT16K_RESOURCES_DIR' in os.environ:
            base_dir = os.environ['AT16K_RESOURCES_DIR']
        else:
            sys_home_dir = str(Path.home())
            base_dir = os.path.join(sys_home_dir, '.at16k')
        model_dir = os.path.join(base_dir, name)
        assert os.path.exists(
            model_dir), ('%s model does not exist at %s' % (name, model_dir))
        return model_dir

    @staticmethod
    def _load_hparams(model_dir):
        params_path = os.path.join(model_dir, 'hparams.json')
        params = json.load(open(params_path, 'r'))
        return params

    @staticmethod
    def _load_vocab(model_dir):
        vocab_path = os.path.join(model_dir, 'bpe.model')
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(vocab_path)
        return sp_model

    def _reset_context_greedy(self):
        _context = {}
        _inferrer = self._inferrer
        _params = self._params
        _context['feats_so_far'] = None
        _context['last_frame_processed'] = 0
        _context['symbols'] = [_params['vocab_null_id']]
        _context['last_a_state'] = np.zeros(
            (_params['audio_encoder_layers'], 2, 1, _params['audio_encoder_units']))
        _last_t_out, _last_t_state = _inferrer.infer_text_encoder(
            inputs=[_context['symbols']], inputs_lengths=[1])
        _context['last_t_out'] = _last_t_out
        _context['last_t_state'] = _last_t_state
        return _context

    def _reset_context_beam(self):
        _context = {}
        _inferrer = self._inferrer
        _params = self._params
        _context['feats_so_far'] = None
        _context['last_frame_processed'] = 0
        _context['last_a_state'] = np.zeros(
            (_params['audio_encoder_layers'], 2, 1, _params['audio_encoder_units']))
        t_state = np.zeros(
            (_params['text_encoder_layers'], 2, 1, _params['text_encoder_units']))
        _context['candidates'] = [
            BeamCandidate(seq=None, hidden=t_state, null_id=_params['vocab_null_id'])]

        return _context

    def _reset_context(self):
        if self._faster:
            return self._reset_context_greedy()
        return self._reset_context_beam()

    def _do_greedy_search(self, _a_out, _params, context):
        _inferrer = self._inferrer
        while True:
            _logits, _probs = _inferrer.infer_joint_encoder(
                a_inputs=_a_out, t_inputs=context['last_t_out'])
            _probs = np.squeeze(_probs)
            _symbol = np.argmax(_probs)
            if _symbol == _params['vocab_null_id']:
                break
            if _symbol == _params['vocab_eos_id']:
                break
            context['symbols'].append(int(_symbol))
            _t_out, _t_state = _inferrer.infer_text_encoder(
                inputs=[[context['symbols'][-1]]], inputs_lengths=[1], inputs_states=context['last_t_state'])
            context['last_t_out'] = _t_out
            context['last_t_state'] = _t_state
        return context

    def _do_beam_search(self, _a_out, _params, context):
        beam_width = self._num_beams
        _inferrer = self._inferrer
        prefix_candidates = list(context['candidates'])
        candidates = []
        loop_num = 0
        while True:
            loop_num += 1
            if not prefix_candidates:
                break
            y_hat = max(prefix_candidates, key=lambda a: a.log_prob)
            prefix_candidates.remove(y_hat)
            _t_out, _t_state = _inferrer.infer_text_encoder(
                inputs=[[y_hat.preds[-1]]], inputs_lengths=[1], inputs_states=y_hat.text_state)
            _logits, _probs = _inferrer.infer_joint_encoder(
                a_inputs=_a_out, t_inputs=_t_out)

            _probs = np.squeeze(_probs)
            candidate = BeamCandidate(y_hat)
            candidate.log_prob += _probs[_params['vocab_null_id']]
            candidates.append(candidate)
            pruned = np.where(_probs > np.log(1e-3))[0]

            for k in pruned:
                if k == _params['vocab_null_id']:
                    continue
                candidate = BeamCandidate(y_hat)
                candidate.log_prob += _probs[k]
                candidate.text_state = _t_state
                candidate.preds.append(int(k))
                candidate.text_outs.append(_t_out)
                prefix_candidates.append(candidate)
            best_score = -np.inf
            if prefix_candidates:
                y_hat = max(prefix_candidates, key=lambda a: a.log_prob)
                best_score = y_hat.log_prob

            scores = [a.log_prob for a in candidates]
            scores.sort(reverse=True)
            scores = scores[:beam_width]
            min_score = min(scores)
            if (len(candidates) >= beam_width and min_score >= best_score) or (loop_num > beam_width):
                break
        candidates = sorted(candidates, key=lambda a: a.log_prob / len(a.preds), reverse=True)
        candidates = candidates[:beam_width]
        context['candidates'] = candidates
        return context

    def __call__(self, samples, context=None):
        _params = self._params
        _inferrer = self._inferrer
        _vocab = self._vocab

        if context is None:
            context = self._reset_context()
        _feats = _inferrer.infer_features(samples)
        if context['feats_so_far'] is None:
            context['feats_so_far'] = _feats
        else:
            context['feats_so_far'] = np.concatenate(
                [context['feats_so_far'], _feats], axis=1)
        _start = context['last_frame_processed']
        _delta_feats = _inferrer.infer_delta_features(context['feats_so_far'])

        # Defining a couple of constants. These are derived from the CNN layer applied over the delta features.
        _window_size = 15
        _step_size = 3

        while True:
            if (_start + _window_size) > _delta_feats.shape[1]:
                break
            _w_feats = _delta_feats[:, _start:(_start + _window_size), :, :]
            _a_out, _a_state = _inferrer.infer_audio_encoder(
                inputs=_w_feats, inputs_states=context['last_a_state'])
            if self._faster:
                context = self._do_greedy_search(_a_out, _params, context)
            else:
                context = self._do_beam_search(_a_out, _params, context)
            context['last_a_state'] = _a_state
            _start += _step_size
        context['last_frame_processed'] = _start
        if self._faster:
            _text = _vocab.DecodeIds(context['symbols'][1:])
        else:
            _candidates = sorted(context['candidates'], key=lambda a: a.log_prob / len(a.preds), reverse=True)
            _top_candidate = _candidates[0]
            _text = _vocab.DecodeIds(_top_candidate.preds[1:])
        return _text, context

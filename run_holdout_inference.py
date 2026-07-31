#!/usr/bin/env python3
# =============================================================================
# run_holdout_inference.py
# -----------------------------------------------------------------------------
# Evaluates all trained model variants on the FROZEN 2024 holdout .mat set
# Generates artifacts: AUC, GT-present MAE, TPR, Trainable-parameter tables
# and ROC and pred-vs-actual plots.
#
# REPRODUCIBILITY FORMAT
#   * inference only: torch.no_grad, model.eval, deterministic algorithms
#   * .mat files are read in sorted order; no RNG anywhere in the eval path
#   * run_info.json records package versions, checkpoint SHA256 hashes, file
#     counts, and the exact CLI invocation
#   * per-sample predictions are exported per model/duration so every number
#     in every table can be recomputed from CSVs alone
#
# MUST SUPPLY (cannot be reconstructed from outside the repo):
#   1. checkpoints_manifest.json  - path to each model x duration checkpoint
#      (template auto-written on first run; fill it in)
#   2. the repo's model-definition .py files, importable from WORKDIR (put
#      them next to this script). Adapters import YOUR classes; only ResNet
#      has a built-in fallback (transcribed from the training notebook).
#   3. optional seed checkpoints for R1-C8: files named like
#      weights/resnet_15_seed2.torchscript are picked up automatically and
#      aggregated into the seed-spread table (train them first with the
#      leak-free notebook, then export).
# =============================================================================
import os, sys, json, glob, hashlib, importlib, platform

# =============================================================================
# CONFIG - edit here 
# =============================================================================
MAT_DIR            = './holdout_2024_matfiles'
CKPT_MANIFEST      = './checkpoints_manifest.json'
OUT_DIR            = './holdout_inference'
MODELS             = ['ResNet', 'CNN-LSTM', 'ConvLSTM', 'TimesNet', 'EQTransformer']


PROBE              = False   # True: load each checkpoint, print I/O format, exit
INSPECT_CHECKPOINT = None    # path: print a state_dict's keys and exit
PREDICT_CHUNK      = 32      # batch size for standard adapters (memory guard)

# =============================================================================
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score, roc_curve, auc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.package import PackageImporter

SR = 100
DURATIONS = (15, 30, 60, 100)
CLASS_NAMES = {0: 'N', 1: 'N+P', 2: 'N+P+S'}
PAPER_MODEL_ORDER = ['ResNet', 'CNN-LSTM', 'ConvLSTM', 'TimesNet', 'EQTransformer']

# ----------------------------------------------------------------------------
# 0. DATA: the frozen holdout
# ----------------------------------------------------------------------------
def load_holdout(mat_dir):
    """Return {duration: list of dicts}, read in sorted (deterministic) order."""
    files = sorted(glob.glob(os.path.join(mat_dir, '*.mat')))
    assert files, f'no .mat files under {mat_dir}'
    data = defaultdict(list)
    for fp in files:
        m = loadmat(fp, squeeze_me=True)
        data[int(m['duration_s'])].append({
            'file': os.path.basename(fp),
            'trace_name': str(m['trace_name']),
            'x': np.asarray(m['waveform'], dtype=np.float32),      # (3, L)
            'label': int(m['label']),
            'p_true': float(m['p_time']),                          # -1 = absent
            's_true': float(m['s_time']),
        })
    for d, lst in sorted(data.items()):
        cnt = defaultdict(int)
        for s in lst:
            cnt[s['label']] += 1
        print(f'  {d:>3d}s: {len(lst)} samples '
              f'(N={cnt[0]}, N+P={cnt[1]}, N+P+S={cnt[2]})')
    return data

# ----------------------------------------------------------------------------
# 1. ADAPTERS - one per architecture; common:
#    load(duration) -> model or None (variant unavailable)
#    predict(model, x_batch (B,3,L)) -> dict with keys
#       p_prob (B,), s_prob (B,), pred_class (B,), p_time (B,), s_time (B,)
#    All probabilities follow the convention:
#       P-exists prob = P(class 1) + P(class 2);  S-exists prob = P(class 2)
# ----------------------------------------------------------------------------
def try_import(module, attr):
    try:
        return getattr(importlib.import_module(module), attr)
    except Exception:
        return None

def load_any(ckpt, device, class_loader=None):
    """TorchScript first (architecture embedded in the file - the HF repo
    ships .torchscript archives); fall back to class + state_dict."""
    try:
        return torch.jit.load(ckpt, map_location=device).to(device).eval()
    except Exception:
        if class_loader is None:
            raise
    model = class_loader()
    state = torch.load(ckpt, map_location=device)
    state = state.get('state_dict', state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()

class ResNetAdapter:
    """Feedforward dual-head ResNet."""
    name = 'ResNet'
    durations = DURATIONS

    class _ResidualBlock1D(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
            self.bn1 = nn.BatchNorm1d(cout)
            self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
            self.bn2 = nn.BatchNorm1d(cout)
            self.shortcut = (nn.Sequential(nn.Conv1d(cin, cout, 1),
                                           nn.BatchNorm1d(cout))
                             if cin != cout else nn.Identity())
        def forward(self, x):
            idn = self.shortcut(x)
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.bn2(self.conv2(out))
            return F.relu(out + idn, inplace=True)

    def _fallback_model(self, L):
        RB = self._ResidualBlock1D
        class ResNet1D(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(3, 64, 7, padding=3)
                self.bn1 = nn.BatchNorm1d(64)
                self.pool = nn.MaxPool1d(4, 4)
                self.stage1 = nn.Sequential(RB(64, 128), RB(128, 128), RB(128, 128))
                self.stage2 = nn.Sequential(RB(128, 256), RB(256, 256), RB(256, 256))
                self.stage3 = nn.Sequential(RB(256, 512), RB(512, 512), RB(512, 512))
                self.stage4 = nn.Sequential(RB(512, 1024), RB(1024, 1024), RB(1024, 1024))
                self.adapt_pool = nn.AdaptiveAvgPool1d(1)
                self.fc_regression = nn.Linear(1024, 2)
                self.fc_classification = nn.Linear(1024, 3)
            def forward(self, x):
                x = self.pool(F.relu(self.bn1(self.conv1(x))))
                for st in (self.stage1, self.stage2, self.stage3, self.stage4):
                    x = self.pool(st(x))
                x = torch.flatten(self.adapt_pool(x), 1)
                reg = self.fc_regression(x)
                p, s = reg.split(1, dim=1)
                return torch.relu(p), torch.relu(s), self.fc_classification(x)
        return ResNet1D()

    def load(self, duration, ckpt, device):
        L = duration * SR
        def class_loader():
            cls = try_import('resnet_model', 'ResNet1D')
            return cls(channels=3, seq_length=L) if cls else self._fallback_model(L)
        return load_any(ckpt, device, class_loader)

    @torch.no_grad()
    def predict(self, model, xb):
        p_hat, s_hat, logits = model(xb)
        prob = F.softmax(logits, dim=1)
        return {'p_prob': (prob[:, 1] + prob[:, 2]).cpu().numpy(),
                's_prob': prob[:, 2].cpu().numpy(),
                'pred_class': prob.argmax(1).cpu().numpy(),
                'p_time': p_hat.squeeze(-1).cpu().numpy(),
                's_time': s_hat.squeeze(-1).cpu().numpy()}

class WindowedAdapter:
    """CNN-LSTM 
      input : [1, num_windows, 3, 1, 900]; windows cut with stride 300 from the
              per-record zero-centered signal 
      output: [1, num_windows, 4] rows = [p_idx, s_idx, p_conf, s_conf] where
              p_idx / s_idx are GLOBAL sample indices 
      aggregation: times from the max-confidence window with conf > 0.8
              among valid (idx >= 0) windows; detection = any window with
              idx >= 0 and conf > 0.5; class masking 1N / 2NP / 2NPS suppresses
              S (and P for noise) when not detected.
      record-level AUC score :
              max conf over valid windows, 0 if none."""
    W, H = 900, 300
    CONF_TIME, CONF_DET = 0.8, 0.5

    def __init__(self, name, module, cls_name):
        self.name, self.module, self.cls_name = name, module, cls_name
        self.durations = DURATIONS

    def load(self, duration, ckpt, device):
        def class_loader():
            cls = try_import(self.module, self.cls_name)
            if cls is None:
                raise ImportError(f'{self.name}: .torchscript load failed and '
                                  f'{self.module}.py not found next to script')
            return cls()
        return load_any(ckpt, device, class_loader)

    @staticmethod
    def _zero_center(acc_nt3):
        """zero_center_normalize, data [N, 3],
        mean over axis=-1 (keepdims), subtracted."""
        return acc_nt3 - acc_nt3.mean(axis=-1, keepdims=True)

    def _prepare(self, x_3n):
        """prepare_cnn_lstm_input: -> [1, n, 3, 1, W] float32 tensor."""
        acc = self._zero_center(x_3n.T.astype(np.float32))          # [N, 3]
        num_steps = acc.shape[0] - self.W + 1
        wins = [acc[s:s + self.W, :].T[:, None, :]                  # [3, 1, W]
                for s in range(0, max(num_steps, 1), self.H)
                if s + self.W <= acc.shape[0]]
        if not wins:
            wins = [np.zeros((3, 1, self.W), dtype=np.float32)]
        return torch.tensor(np.stack(wins))[None]                   # [1, n, 3, 1, W]

    @torch.no_grad()
    def predict(self, model, xb):
        res = defaultdict(list)
        self._win_cache = []                 # per-record window scores (wAUC)
        for x in xb:
            wins = self._prepare(x.cpu().numpy()).to(xb.device)
            pred = model(wins).squeeze(0).cpu()                     # [n, 4]
            p_idx, s_idx = pred[:, 0], pred[:, 1]
            p_conf, s_conf = pred[:, 2], pred[:, 3]
            self._win_cache.append({
                'starts': np.arange(pred.shape[0]) * float(self.H),
                'W': float(self.W),
                'p': np.clip(p_conf.numpy(), 0, 1),
                's': np.clip(s_conf.numpy(), 0, 1)})

            def agg(idx, conf):
                valid = (idx >= 0) & (conf > self.CONF_TIME)
                if valid.any():
                    best = torch.argmax(conf[valid])
                    t = float(idx[valid][best].item()) / SR
                else:
                    t = -1.0
                score_valid = idx >= 0
                if score_valid.any():
                    score = float(conf[score_valid].max().item())
                    best_raw = torch.argmax(conf[score_valid])
                    t_raw = float(idx[score_valid][best_raw].item()) / SR
                else:
                    score, t_raw = 0.0, -1.0
                detected = bool(((idx >= 0) & (conf > self.CONF_DET)).any().item())
                return t, t_raw, score, detected

            p_t, p_raw, p_score, p_det = agg(p_idx, p_conf)
            s_t, s_raw, s_score, s_det = agg(s_idx, s_conf)
            cls = 2 if (p_det and s_det) else (1 if p_det else 0)
            # masking: S only reported for 2NPS, P suppressed for 1N
            res['p_prob'].append(p_score)
            res['s_prob'].append(s_score)
            res['pred_class'].append(cls)
            res['p_time'].append(p_t if cls >= 1 else -1.0)
            res['s_time'].append(s_t if cls == 2 else -1.0)
            res['p_time_raw'].append(p_raw)      # head output, no class masking
            res['s_time_raw'].append(s_raw)
        return {k: np.asarray(v, dtype=float) for k, v in res.items()}

class StandardAdapter:
    """Two-phase TimesNet (frozen encoder + detection heads). Requires the
    model file; expected forward: x (B,3,L) -> (p_time, s_time,
    p_logit, s_logit) with independent binary heads."""
    durations = DURATIONS

    def __init__(self, name, fb_module=None, fb_cls=None):
        self.name, self.fb_module, self.fb_cls = name, fb_module, fb_cls

    def load(self, duration, ckpt, device):
        def class_loader():
            cls = try_import(self.fb_module, self.fb_cls) if self.fb_module else None
            if cls is None:
                raise ImportError(f'{self.name}: .torchscript load failed and '
                                  f'no fallback class module available')
            return cls()
        return load_any(ckpt, device, class_loader)

    @torch.no_grad()
    def predict(self, model, xb):
        chunks = [self._predict_chunk(model, xb[i:i + PREDICT_CHUNK])
                  for i in range(0, len(xb), PREDICT_CHUNK)]
        return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}

    @staticmethod
    def _vec(a, n):
        """Broadcast to a length-n 1-D array"""
        a = np.atleast_1d(np.asarray(a, dtype=float))
        return a if a.shape == (n,) else np.broadcast_to(a, (n,)).copy()

    @torch.no_grad()
    def _predict_chunk(self, model, xb):
        out = model(xb)
        if len(out) == 4:                 # binary P/S heads (2-phase setup)
            p_hat, s_hat, p_logit, s_logit = out
            pp = torch.sigmoid(p_logit).squeeze(-1).cpu().numpy()
            sp = torch.sigmoid(s_logit).squeeze(-1).cpu().numpy()
            pred = np.where(sp >= .5, 2, np.where(pp >= .5, 1, 0))
        elif len(out) == 3:               # ResNet-style 3-class head
            p_hat, s_hat, logits = out
            prob = F.softmax(logits, dim=1)
            pp = (prob[:, 1] + prob[:, 2]).cpu().numpy()
            sp = prob[:, 2].cpu().numpy()
            pred = prob.argmax(1).cpu().numpy()
        else:
            raise RuntimeError(f'{self.name}: unexpected {len(out)}-tuple '
                               f'output; set PROBE=True and adapt predict()')
        n = xb.shape[0]
        return {'p_prob': self._vec(pp, n), 's_prob': self._vec(sp, n),
                'pred_class': self._vec(pred, n),
                'p_time': self._vec(p_hat.squeeze(-1).cpu().numpy(), n),
                's_time': self._vec(s_hat.squeeze(-1).cpu().numpy(), n)}

class ConvLSTMAdapter:
    """ConvLSTM
      preprocessing : zero-center normalize (mean over the last axis of the
                      [N, 3] array, subtracted), then windowing with
                      DURATION-SPECIFIC parameters (see WINDOWING)
      input         : [1, num_windows, 3, 1, window_size]
      output        : [1, num_windows, 6] rows = [p_idx, s_idx, logits(4)],
                      indices WINDOW-LOCAL, negative = invalid;
                      classes map {0: N, 1: N+P, 2: N+P+S, 3: N+P+S}
      aggregation   : arrival time = mean over valid (idx >= 0) windows of
                      (window_start + local_idx);
                      class = argmax of window-averaged logits; S reported
                      only for N+P+S, P suppressed for noise (their masking)
      device        : forced to CPU, as in the provider's runner
    Record-level AUC scores (harness addition, monotone with their rule):
      softmax of the window-averaged logits; P-exists = P(1)+P(2)+P(3),
      S-exists = P(2)+P(3)."""
    name = 'ConvLSTM'
    durations = DURATIONS
    WINDOWING = {15: (900, 300), 30: (900, 300),
                 60: (3000, 500), 100: (5000, 1000)}

    def load(self, duration, ckpt, device):
        model = torch.jit.load(ckpt, map_location='cpu').eval()
        self._wh = self.WINDOWING[duration]
        return model

    @staticmethod
    def _zero_center(acc_nt3):
        return acc_nt3 - acc_nt3.mean(axis=-1, keepdims=True)

    def _prepare(self, x_3n):
        W, H = self._wh
        acc = self._zero_center(x_3n.T.astype(np.float32))          # [N, 3]
        wins = []
        num_steps = acc.shape[0] - W + 1
        for start in range(0, max(num_steps, 1), H):
            end = start + W
            if end > acc.shape[0]:
                break
            wins.append(acc[start:end, :].T[:, None, :])            # [3, 1, W]
        if not wins:
            wins = [np.zeros((3, 1, W), dtype=np.float32)]
        return torch.tensor(np.stack(wins))[None]                   # [1, n, 3, 1, W]

    @torch.no_grad()
    def predict(self, model, xb):
        res = defaultdict(list)
        self._win_cache = []                 # per-record window scores (wAUC)
        for x in xb:
            wins = self._prepare(x.cpu().numpy())                   # CPU
            pred = model(wins).squeeze(0)                           # [n, 6]
            p_idx, s_idx = pred[:, 0], pred[:, 1]
            logits = pred[:, 2:6]

            sm = F.softmax(logits, dim=1)                           # per window
            W, Hh = self._wh
            self._win_cache.append({
                'starts': np.arange(pred.shape[0]) * float(Hh), 'W': float(W),
                'p': (sm[:, 1] + sm[:, 3]).cpu().numpy(),
                's': (sm[:, 2] + sm[:, 3]).cpu().numpy()})

            prob = F.softmax(logits.mean(dim=0), dim=0)             # averaged logits
            cls4 = int(torch.argmax(prob).item())
            cls = {0: 0, 1: 1, 2: 2, 3: 2}[cls4]
            p_prob = float(prob[1] + prob[2] + prob[3])
            s_prob = float(prob[2] + prob[3])

            starts = torch.arange(pred.shape[0], dtype=torch.float32) * Hh

            def mean_valid(idx):
                m = idx >= 0
                if not m.any():
                    return -1.0
                return float((starts[m] + idx[m]).mean().item()) / SR
            p_t, s_t = mean_valid(p_idx), mean_valid(s_idx)

            res['p_prob'].append(p_prob)
            res['s_prob'].append(s_prob)
            res['pred_class'].append(cls)
            res['p_time'].append(p_t if cls >= 1 else -1.0)         # masking
            res['s_time'].append(s_t if cls == 2 else -1.0)
            res['p_time_raw'].append(p_t)        # head output, no class masking
            res['s_time_raw'].append(s_t)
        return {k: np.asarray(v, dtype=float) for k, v in res.items()}

class EQTransformerAdapter:
    """Pretrained EQTransformer via SeisBench, 60 s inputs only. Optional: skipped if seisbench missing.
    """
    name = 'EQTransformer'
    durations = (60,)

    def load(self, duration, ckpt, device):
        import seisbench.models as sbm                     # noqa
        model = sbm.EQTransformer.from_pretrained('original')
        return model.to(device).eval()

    @torch.no_grad()
    def predict(self, model, xb):
        res = defaultdict(list)
        L = int(model.in_samples)              # 6000 for 'original' weights
        for x in xb:
            arr = x.cpu().numpy()
            wav = np.zeros((3, L), dtype=np.float32)
            n = min(L, arr.shape[1])
            wav[:, :n] = arr[:, :n]
            # SeisBench annotate-style preprocessing for EQT 'original':
            # demean per channel, divide by the window's GLOBAL std
            wav = wav - wav.mean(axis=1, keepdims=True)
            wav = wav / (wav.std() + 1e-10)
            t = torch.tensor(wav[None], device=x.device)
            det, p, s = model(t)                           # (1, T) each
            det, p, s = (v.squeeze().cpu().numpy() for v in (det, p, s))
            res['p_prob'].append(float(p.max()))
            res['s_prob'].append(float(s.max()))
            res['pred_class'].append(2 if s.max() >= .3 else (1 if p.max() >= .3 else 0))
            res['p_time'].append(float(np.argmax(p)) / SR)
            res['s_time'].append(float(np.argmax(s)) / SR)
        return {k: np.asarray(v) for k, v in res.items()}

class TimesNetAdapter:
    """Duration-specific TimesNet P/S torch.package exports.
    """
    name = 'TimesNet'
    durations = DURATIONS

    def load(self, duration, ckpt, device):
        errs = []
        try:                                        # (a) torch.package
            m = PackageImporter(ckpt).load_pickle('model', 'model.pkl')
            return m.to('cpu').eval()
        except Exception as e:
            errs.append(f'package: {type(e).__name__}: {str(e)[:90]}')
        try:                                        # (b) TorchScript
            return torch.jit.load(ckpt, map_location='cpu').eval()
        except Exception as e:
            errs.append(f'jit: {type(e).__name__}: {str(e)[:90]}')
        try:                                        # (c) plain torch.save
            obj = torch.load(ckpt, map_location='cpu', weights_only=False)
        except (ModuleNotFoundError, AttributeError) as e:
            raise RuntimeError(
                f'{os.path.basename(ckpt)} is a plain torch.save of the full '
                f'model and unpickling needs the original class ({e}). Put '
                f'the model-definition .py from the model owner next to this '
                f'script and rerun.') from e
        except Exception as e:
            errs.append(f'torch.load: {type(e).__name__}: {str(e)[:90]}')
            raise RuntimeError('unrecognized TimesNet file format; attempts: '
                               + ' | '.join(errs))
        if isinstance(obj, nn.Module):
            return obj.to('cpu').eval()
        if isinstance(obj, dict):
            inner = obj.get('model')
            if isinstance(inner, nn.Module):
                return inner.to('cpu').eval()
            keys = list(obj)[:8]
            raise RuntimeError(
                f'{os.path.basename(ckpt)} is a checkpoint dict with keys '
                f'{keys} - a bare state_dict cannot be run without the model ')
        raise RuntimeError(f'{os.path.basename(ckpt)}: torch.load returned '
                           f'{type(obj).__name__}; attempts: ' + ' | '.join(errs))

    @staticmethod
    def _to_prob(v):
        v = float(v)
        return v if 0.0 <= v <= 1.0 else float(1.0 / (1.0 + np.exp(-v)))

    @torch.no_grad()
    def predict(self, model, xb):
        dur = xb.shape[-1] / SR
        res = defaultdict(list)
        fn = model.timesnet if hasattr(model, 'timesnet') else model
        for x in xb:
            x_enc = x.cpu().T.contiguous().unsqueeze(0)        # (1, T, 3)
            out = fn(x_enc)
            if len(out) == 4:              # package contract
                p_prob, s_prob, p_raw, s_raw = out
            elif len(out) == 2:            # raw network: (reg[1,2], cls[1,2])
                reg, cls = out
                reg, cls = reg.reshape(-1), cls.reshape(-1)
                p_prob, s_prob = cls[0], cls[1]
                p_raw, s_raw = reg[0], reg[1]
            else:
                raise RuntimeError(f'TimesNet: unexpected {len(out)}-tuple output')
            p_prob = self._to_prob(p_prob.reshape(-1)[0])
            s_prob = self._to_prob(s_prob.reshape(-1)[0])
            def t_ok(v):
                v = float(v)
                return v if 0.0 <= v <= dur * 1.05 else -1.0
            p_t, s_t = t_ok(p_raw.reshape(-1)[0]), t_ok(s_raw.reshape(-1)[0])
            c = 2 if (p_prob >= .5 and s_prob >= .5) else (1 if p_prob >= .5 else 0)
            res['p_prob'].append(p_prob)
            res['s_prob'].append(s_prob)
            res['pred_class'].append(c)
            res['p_time'].append(min(p_t, dur) if (c >= 1 and p_t >= 0) else -1.0)
            res['s_time'].append(min(s_t, dur) if (c == 2 and s_t >= 0) else -1.0)
            res['p_time_raw'].append(min(p_t, dur) if p_t >= 0 else -1.0)
            res['s_time_raw'].append(min(s_t, dur) if s_t >= 0 else -1.0)
        return {k: np.asarray(v, dtype=float) for k, v in res.items()}

ADAPTERS = {
    'ResNet':        ResNetAdapter(),
    'CNN-LSTM':      WindowedAdapter('CNN-LSTM', 'cnn_lstm_model', 'CNNLSTM'),
    'ConvLSTM':      ConvLSTMAdapter(),
    'TimesNet':      TimesNetAdapter(),
    'EQTransformer': EQTransformerAdapter(),
}

# default checkpoint filenames (used when the manifest has no entry) - these match the HuggingFace naming in weights/
CKPT_STEM = {'ResNet': 'resnet', 'CNN-LSTM': 'cnn_lstm',
             'ConvLSTM': 'convlstm', 'TimesNet': 'timesnet'}

def resolve_ckpts(ckpts, mname, dur):
    """Return the list of checkpoints for (model, duration): the canonical
    one first"""
    if mname == 'TimesNet':
        # accept the package in torchscript
        candidates = [ckpts.get('TimesNet', {}).get(str(dur)),
                      os.path.join('weights', f'timesnet_{dur}.torchscript')]
        base = next((c for c in candidates if c and os.path.exists(c)),
                    candidates[-1])
        cks = [base] if base else []
        if base and os.path.exists(base):
            root, ext = os.path.splitext(base)
            cks += sorted(glob.glob(f'{root}_seed*{ext}'))
        return cks
    entry = ckpts.get(mname, {}).get(str(dur))
    if isinstance(entry, list):
        return [p for p in entry if p and os.path.exists(p)] or entry[:1]
    cks = []
    if entry and os.path.exists(entry):
        cks.append(entry)
    else:
        default = os.path.join(
            'weights', f'{CKPT_STEM.get(mname, mname.lower())}_{dur}.torchscript')
        if os.path.exists(default):
            cks.append(default)
        elif entry:
            cks.append(entry)
    return cks

# ----------------------------------------------------------------------------
# 2. METRICS - definitions
#    Table 1: AUC of P/S EVENT DETECTION (P-exists, S-exists)
#    Table 2: MAE (s) over ALL samples where the phase is PRESENT IN GROUND
#             TRUTH, using the raw (class-unmasked) regression output -
#             detection is NOT a condition; detection quality is shown by AUC
#             and the TPR tables. Samples where the model emits no
#             valid time (raw = -1) are excluded by necessity; counts are
#             reported (n_p_eval / n_s_eval vs n_p_gt / n_s_gt).
#    TPR:     true-positive rate of the class decision, kept as companion.
# ----------------------------------------------------------------------------
def evaluate_window_auc(win_cache, samples, n_thresholds=100):
    """Window-level tolerant AUC, replicating the ORIGINAL window-model
    evaluation protocol: a positive window score up to `tol` windows AFTER
    the true window counts as a hit. Companion metric only - the record-level
    Table 1 is computed independently and is unchanged."""
    labs, scores = {'p': [], 's': []}, {'p': [], 's': []}
    for wc, smp in zip(win_cache, samples):
        starts, W = wc['starts'], wc['W']
        for ph, t_true in (('p', smp['p_true']), ('s', smp['s_true'])):
            lab = ((t_true >= 0) &
                   (t_true * SR >= starts) &
                   (t_true * SR < starts + W)).astype(int)
            labs[ph].append(lab)
            scores[ph].append(wc[ph])
    out = {}
    thresholds = np.linspace(0, 1, n_thresholds)
    for ph in ('p', 's'):
        tprs, fprs = [], []
        for th in thresholds:
            TP = FP = FN = TN = 0
            for sc, lb in zip(scores[ph], labs[ph]):
                pb = (sc >= th).astype(int)
                for i in range(len(lb)):
                    if lb[i] == 1:
                        if pb[i:i + 1].any():
                            TP += 1
                        else:
                            FN += 1
                    else:
                        if pb[i] == 1:
                            FP += 1
                        else:
                            TN += 1
            tprs.append(TP / (TP + FN) if TP + FN else 0.0)
            fprs.append(FP / (FP + TN) if FP + TN else 0.0)
        order = np.argsort(fprs)
        _trap = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
        out[f'{ph}_wauc'] = float(_trap(np.asarray(tprs)[order],
                                        np.asarray(fprs)[order]))
    return out

def evaluate(df):
    y, yp = df.y_true.values, df.pred_class.values
    p_gt = (y >= 1)
    s_gt = (y == 2)
    p_eval = p_gt & (df.p_pred_raw.values >= 0)
    s_eval = s_gt & (df.s_pred_raw.values >= 0)
    p_hit = p_gt & (yp >= 1)
    s_hit = s_gt & (yp == 2)
    out = {
        'p_auc': roc_auc_score(p_gt.astype(int), df.p_prob),
        's_auc': roc_auc_score(s_gt.astype(int), df.s_prob),
        'p_mae': float(np.abs(df.p_pred_raw[p_eval] - df.p_true[p_eval]).mean()) if p_eval.any() else np.nan,
        's_mae': float(np.abs(df.s_pred_raw[s_eval] - df.s_true[s_eval]).mean()) if s_eval.any() else np.nan,
        'p_hit_rate': float(p_hit.sum() / max(p_gt.sum(), 1)),
        's_hit_rate': float(s_hit.sum() / max(s_gt.sum(), 1)),
        'n': len(df),
        'n_p_gt': int(p_gt.sum()), 'n_s_gt': int(s_gt.sum()),
        'n_p_eval': int(p_eval.sum()), 'n_s_eval': int(s_eval.sum()),
    }
    return out

def fmt(v, nd=2):
    return 'NA' if (v is None or (isinstance(v, float) and np.isnan(v))) else f'{v:.{nd}f}'

def paper_tables(all_metrics, out_dir):
    """Create Tables in the manuscript layout (markdown +
    LaTeX) plus the hit-rate companion table (new)."""
    def grid(key, nd):
        rows = []
        for m in PAPER_MODEL_ORDER:
            row = {'Model': m}
            for d in DURATIONS:
                met = all_metrics.get((m, d))
                row[f'{d}s P'] = fmt(met[f'p_{key}'] if met else None, nd)
                row[f'{d}s S'] = fmt(met[f's_{key}'] if met else None, nd)
            rows.append(row)
        return pd.DataFrame(rows).set_index('Model')

    t1 = grid('auc', 2)
    t2 = grid('mae', 2)
    t3 = grid('hit_rate', 2)
    with open(os.path.join(out_dir, 'paper_tables.md'), 'w') as fh:
        fh.write('## Table 1 replacement - AUC of P and S event detection '
                 '(2024 holdout)\n\n' + t1.to_markdown() + '\n\n')
        fh.write('## Table 2 replacement - MAE (s) of arrival times over all '
                 'samples with the phase present in ground truth (2024 '
                 'holdout; detection not conditioned)\n\n'
                 + t2.to_markdown() + '\n\n')
        fh.write('## Companion table - detection TPR (true positive rate; '
                 'report next to Table 2, addresses Reviewer 1, M2)\n\n'
                 + t3.to_markdown() + '\n')
    for name, t in (('table1_auc', t1), ('table2_mae', t2), ('table3_hitrate', t3)):
        t.to_csv(os.path.join(out_dir, f'{name}.csv'))
        with open(os.path.join(out_dir, f'{name}.tex'), 'w') as fh:
            fh.write(t.to_latex())
    return t1, t2, t3


# ----------------------------------------------------------------------------
# 3. MAIN
# ----------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    if INSPECT_CHECKPOINT:
        st = torch.load(INSPECT_CHECKPOINT, map_location='cpu')
        st = st.get('state_dict', st) if isinstance(st, dict) else st
        for k, v in st.items():
            print(f'{k:60s} {tuple(v.shape)}')
        return

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(CKPT_MANIFEST):

        needs_manifest = any(
            model not in ('TimesNet', 'EQTransformer') for model in MODELS)
        if not needs_manifest:
            ckpts = {}
        else:
            tmpl = {m: {str(d): f'weights/{m.lower().replace("-", "_")}_{d}s.pt'
                        for d in ADAPTERS[m].durations if m != 'EQTransformer'}
                    for m in ADAPTERS}
            tmpl['EQTransformer'] = {'60': 'seisbench:original (auto)'}
            tmpl['TimesNet'] = {
                str(d): os.path.join(
                    'exports', 'torchscript', f'timesnet_{d}s.pt')
                for d in ADAPTERS['TimesNet'].durations
            }
            with open(CKPT_MANIFEST, 'w') as fh:
                json.dump(tmpl, fh, indent=1)
            sys.exit(f'wrote template {CKPT_MANIFEST} - fill in checkpoint '
                     f'paths into "weights" folder (from HuggingFace downloads) and rerun')
    else:
        with open(CKPT_MANIFEST) as fh:
            ckpts = json.load(fh)

    if PROBE:
        def desc(o):
            if isinstance(o, torch.Tensor):
                return f'Tensor{tuple(o.shape)}'
            if isinstance(o, (tuple, list)):
                return '(' + ', '.join(desc(x) for x in o) + ')'
            return type(o).__name__
        for mname, durs in ckpts.items():
            if mname == 'EQTransformer':
                continue
            for dur, ck in durs.items():
                if not ck or not os.path.exists(ck):
                    print(f'{mname}/{dur}s: missing ({ck})')
                    continue
                L = int(dur) * SR
                try:
                    model = ADAPTERS[mname].load(int(dur), ck, device)
                except Exception as e:
                    print(f'{mname}/{dur}s: LOAD FAILED - {e}')
                    continue
                for shape in [(1, 3, 3, 1, 900), (2, 3, 1, 900), (1, 3, L), (1, L, 3)]:
                    try:
                        with torch.no_grad():
                            out = model(torch.randn(*shape, device=device))
                        print(f'{mname}/{dur}s: input {shape} -> {desc(out)}')
                        break
                    except Exception:
                        continue
                else:
                    print(f'{mname}/{dur}s: no candidate input shape accepted')
                del model
        return

    print('=== loading frozen holdout ===')
    data = load_holdout(MAT_DIR)

    run_info = {'started': datetime.now().isoformat(timespec='seconds'),
                'argv': sys.argv, 'device': str(device),
                'python': platform.python_version(),
                'torch': torch.__version__, 'numpy': np.__version__,
                'n_mat_files': sum(len(v) for v in data.values()),
                'checkpoints': {}, 'skipped': []}

    pred_frames, all_metrics, seed_rows, params = {}, {}, [], {}
    for mname in MODELS:
        adapter = ADAPTERS[mname]
        for dur in adapter.durations:
            cks = (resolve_ckpts(ckpts, mname, dur)
                   if mname != 'EQTransformer' else ['seisbench'])

            seed_metrics = []
            for si, ck in enumerate(cks):
                tag = f'{mname}/{dur}s' + (f'/seed{si + 1}' if len(cks) > 1 else '')
                try:
                    model = adapter.load(dur, ck, device)
                except Exception as e:
                    run_info['skipped'].append(f'{tag}: {type(e).__name__}: {e}')
                    print(f'SKIP {tag} - {type(e).__name__}: {e}')
                    continue
                if mname != 'EQTransformer':
                    run_info['checkpoints'][tag] = {'path': ck, 'sha256': sha256(ck)}
                try:
                    n_par = int(sum(p.numel() for p in model.parameters()))
                except Exception:
                    n_par = -1
                if si == 0:
                    params[(mname, dur)] = n_par

                samples = data[dur]
                xb = torch.tensor(np.stack([s['x'] for s in samples])).to(device)
                preds = adapter.predict(model, xb)
                # non-masking adapters (ResNet, Standard, EQT): raw == reported
                preds.setdefault('p_time_raw', preds['p_time'])
                preds.setdefault('s_time_raw', preds['s_time'])
                df = pd.DataFrame({
                    'file': [s['file'] for s in samples],
                    'trace_name': [s['trace_name'] for s in samples],
                    'y_true': [s['label'] for s in samples],
                    'p_true': [s['p_true'] for s in samples],
                    's_true': [s['s_true'] for s in samples],
                    'p_prob': preds['p_prob'], 's_prob': preds['s_prob'],
                    'pred_class': preds['pred_class'],
                    'p_pred': preds['p_time'], 's_pred': preds['s_time'],
                    'p_pred_raw': preds['p_time_raw'],
                    's_pred_raw': preds['s_time_raw']})
                suffix = '' if si == 0 else f'_seed{si + 1}'
                csv = os.path.join(OUT_DIR, f'predictions_{mname}_{dur}s{suffix}.csv')
                df.to_csv(csv, index=False)

                m = evaluate(df)
                win = getattr(adapter, '_win_cache', None)
                if win:
                    m.update(evaluate_window_auc(win, samples))
                    adapter._win_cache = None
                wtxt = (f' | P {m["p_wauc"]:.2f} '
                        f'S {m["s_wauc"]:.2f}') if 'p_wauc' in m else ''
                seed_metrics.append(m)
                if si == 0:
                    all_metrics[(mname, dur)] = m
                    pred_frames[(mname, dur)] = df
                stag = f' [seed {si + 1}/{len(cks)}]' if len(cks) > 1 else ''
                print(f'{mname:14s} {dur:>3d}s | AUC P {m["p_auc"]:.2f} S {m["s_auc"]:.2f} '
                      f'| MAE(GT) P {fmt(m["p_mae"])} S {fmt(m["s_mae"])} '
                      f'({m["n_p_eval"]}/{m["n_p_gt"]}, {m["n_s_eval"]}/{m["n_s_gt"]}) '
                      f'| TPR P {m["p_hit_rate"]:.2f} S {m["s_hit_rate"]:.2f}{wtxt} '
                      f'| {n_par / 1e6:.2f}M par{stag} -> {os.path.basename(csv)}')
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            if len(seed_metrics) > 1:
                keys = ('p_auc', 's_auc', 'p_mae', 's_mae',
                        'p_hit_rate', 's_hit_rate')
                agg = dict(all_metrics[(mname, dur)])
                for k in keys:
                    vals = [sm[k] for sm in seed_metrics
                            if not (isinstance(sm[k], float) and np.isnan(sm[k]))]
                    if vals:
                        agg[k] = float(np.mean(vals))
                        seed_rows.append({'model': mname, 'duration_s': dur,
                                          'metric': k, 'n_seeds': len(vals),
                                          'mean': round(float(np.mean(vals)), 3),
                                          'min': round(float(np.min(vals)), 3),
                                          'max': round(float(np.max(vals)), 3)})
                all_metrics[(mname, dur)] = agg   # tables report seed means

    if not all_metrics:
        sys.exit('nothing evaluated - check checkpoints/adapters')

    print('\n=== paper tables ===')
    t1, t2, t3 = paper_tables(all_metrics, OUT_DIR)
    print('\nTable 1 (AUC):');       print(t1.to_string())
    print('\nTable 2 (GT-present MAE s):'); print(t2.to_string())
    print('\nTPR:');                        print(t3.to_string())
    pd.DataFrame([{'model': k[0], 'duration_s': k[1], **v}
                  for k, v in all_metrics.items()]).to_csv(
        os.path.join(OUT_DIR, 'metrics_full.csv'), index=False)


    # trainable parameters
    if params:
        prow = []
        for mn in PAPER_MODEL_ORDER:
            row = {'Model': mn}
            for d in DURATIONS:
                v = params.get((mn, d))
                row[f'{d}s'] = f'{v / 1e6:.2f}M' if v and v > 0 else 'NA'
            prow.append(row)
        tp = pd.DataFrame(prow).set_index('Model')
        print('\nTrainable parameters:')
        print(tp.to_string())
        tp.to_csv(os.path.join(OUT_DIR, 'table_parameters.csv'))

    # per-figure config selection:
    #   fig3 (ROC): max mean(P,S AUC)         - a detection figure
    #   fig4 (regression): min mean hits-MAE among configs with BOTH hit
    #        rates >= 0.5 (floor prevents detection-starved configs from
    #        winning on a handful of easy hits); falls back to all configs
    def pick(metric):
        best = {}
        for (mn, d), met in sorted(all_metrics.items(), key=lambda kv: kv[0][1]):
            sc = metric(met)
            if sc is None:
                continue
            if mn not in best or sc > best[mn][1]:
                best[mn] = (d, sc)
        return {m: d for m, (d, _) in best.items()}

    cfg3 = pick(lambda m: (m['p_auc'] + m['s_auc']) / 2)
    def mae_score(m, floor=0.5):
        if np.isnan(m['p_mae']) or np.isnan(m['s_mae']):
            return None
        if m['p_hit_rate'] < floor or m['s_hit_rate'] < floor:
            return None
        return -(m['p_mae'] + m['s_mae']) / 2
    cfg4 = pick(mae_score)
    cfg4_fallback = pick(lambda m: None if (np.isnan(m['p_mae']) or
                                            np.isnan(m['s_mae']))
                         else -(m['p_mae'] + m['s_mae']) / 2)
    for mn in cfg3:
        cfg4.setdefault(mn, cfg4_fallback.get(mn, cfg3[mn]))


    run_info['finished'] = datetime.now().isoformat(timespec='seconds')
    run_info['best_configs'] = {'fig3': cfg3, 'fig4': cfg4}
    with open(os.path.join(OUT_DIR, 'run_info.json'), 'w') as fh:
        json.dump(run_info, fh, indent=1)
    print(f'\nDONE. Everything under {OUT_DIR}/')

if __name__ == '__main__':
    main()
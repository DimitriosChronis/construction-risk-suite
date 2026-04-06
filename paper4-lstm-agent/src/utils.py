"""
Paper 4 — Shared Utilities
============================
Common LSTM classes, training loops, ensemble logic, and helpers
used across all Paper 4 scripts.

Centralises code that was previously copy-pasted in scripts 02-13.
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ══════════════════════════════════════════════════════════════════════════════
# DEFAULTS (can be overridden by callers)
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_N_LAYERS    = 2
DEFAULT_DROPOUT     = 0.3
DEFAULT_LR          = 5e-4
DEFAULT_EPOCHS      = 150
DEFAULT_BATCH_SIZE  = 16
DEFAULT_PATIENCE    = 20
DEFAULT_SEEDS       = [42, 43, 44, 45, 46]


# ══════════════════════════════════════════════════════════════════════════════
# 1. MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
class LSTMClassifier(nn.Module):
    """LSTM binary classifier for crisis regime prediction.

    Parameters
    ----------
    input_size : int
        Number of input features per timestep.
    hidden_size : int
        LSTM hidden dimension.
    n_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout rate applied after batch-norm.
    use_bn : bool
        If True (default), apply BatchNorm1d after the LSTM.
    """
    def __init__(self, input_size, hidden_size=DEFAULT_HIDDEN_SIZE,
                 n_layers=DEFAULT_N_LAYERS, dropout=DEFAULT_DROPOUT,
                 use_bn=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        if self.use_bn:
            h = self.bn(h)
        return self.fc(self.drop(h)).squeeze()


class RNNClassifier(nn.Module):
    """Flexible RNN classifier supporting LSTM and GRU backends."""
    def __init__(self, input_size, hidden_size=DEFAULT_HIDDEN_SIZE,
                 n_layers=DEFAULT_N_LAYERS, dropout=DEFAULT_DROPOUT,
                 rnn_type="LSTM", use_bn=True):
        super().__init__()
        RNN = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn  = RNN(input_size, hidden_size, n_layers,
                        batch_first=True,
                        dropout=dropout if n_layers > 1 else 0)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        h = out[:, -1, :]
        if self.use_bn:
            h = self.bn(h)
        return self.fc(self.drop(h)).squeeze()


# ══════════════════════════════════════════════════════════════════════════════
# 2. SEQUENCE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════
def make_sequences(X, y, lookback, lead, dates=None):
    """Build (lookback, features) sequences with a lead-time offset.

    Parameters
    ----------
    X : np.ndarray, shape (T, n_features)
    y : np.ndarray, shape (T,)
    lookback : int
    lead : int
    dates : array-like or None
        If provided, returns aligned date indices.

    Returns
    -------
    X_seq : np.ndarray, shape (N, lookback, n_features)
    y_seq : np.ndarray, shape (N,)
    d_seq : np.ndarray or None
    """
    Xs, ys, ds = [], [], []
    for i in range(lookback, len(X) - lead):
        Xs.append(X[i - lookback:i])
        ys.append(y[i + lead])
        if dates is not None:
            ds.append(dates[i + lead])
    X_seq = np.array(Xs)
    y_seq = np.array(ys)
    d_seq = np.array(ds) if dates is not None else None
    return (X_seq, y_seq, d_seq) if dates is not None else (X_seq, y_seq)


def temporal_summary(X_3d):
    """Extract temporal summary from (n, lookback, features) -> (n, features*4).

    Produces per-feature: mean, std, last value, trend (last - first).
    Used to create 2D inputs for sklearn models from 3D RNN sequences.
    """
    mean  = X_3d.mean(axis=1)
    std   = X_3d.std(axis=1)
    last  = X_3d[:, -1, :]
    trend = X_3d[:, -1, :] - X_3d[:, 0, :]
    return np.hstack([mean, std, last, trend])


# ══════════════════════════════════════════════════════════════════════════════
# 3. SINGLE-MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_single_lstm(X_tr, y_tr, seed=42, epochs=DEFAULT_EPOCHS,
                      batch_size=DEFAULT_BATCH_SIZE,
                      hidden_size=DEFAULT_HIDDEN_SIZE,
                      n_layers=DEFAULT_N_LAYERS,
                      dropout=DEFAULT_DROPOUT,
                      lr=DEFAULT_LR, patience=DEFAULT_PATIENCE,
                      use_bn=True, rnn_type="LSTM"):
    """Train a single LSTM/GRU model and return it.

    Parameters
    ----------
    X_tr : np.ndarray, shape (N, lookback, n_features)
    y_tr : np.ndarray, shape (N,)
    seed : int
    rnn_type : str, "LSTM" or "GRU"

    Returns
    -------
    model : nn.Module (in eval mode, on CPU)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    feat_dim = X_tr.shape[2]

    # Class balancing via pos_weight
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    pos_w = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    if rnn_type == "LSTM" and not hasattr(LSTMClassifier, '_is_rnn'):
        model = LSTMClassifier(feat_dim, hidden_size, n_layers, dropout,
                               use_bn=use_bn)
    else:
        model = RNNClassifier(feat_dim, hidden_size, n_layers, dropout,
                              rnn_type=rnn_type, use_bn=use_bn)

    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    ds   = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    dl   = DataLoader(ds, batch_size=batch_size, shuffle=True,
                      drop_last=(len(ds) > batch_size))

    best_loss, best_state, wait = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        ep_loss = 0
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        avg = ep_loss / len(dl)
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.clone() for k, v
                          in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def predict_probs(model, X_te):
    """Get sigmoid probabilities from a trained model.

    Parameters
    ----------
    model : nn.Module
    X_te : np.ndarray, shape (N, lookback, n_features)

    Returns
    -------
    probs : np.ndarray, shape (N,)
    """
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_te)).numpy()
    if logits.ndim == 0:
        logits = logits.reshape(1)
    return 1.0 / (1.0 + np.exp(-logits))


# ══════════════════════════════════════════════════════════════════════════════
# 5. ENSEMBLE TRAINING & PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def train_ensemble(X_tr, y_tr, seeds=None, verbose=False, **kwargs):
    """Train multiple LSTM models with different random seeds.

    Parameters
    ----------
    X_tr : np.ndarray, shape (N, lookback, n_features)
    y_tr : np.ndarray, shape (N,)
    seeds : list of int, default DEFAULT_SEEDS
    verbose : bool
    **kwargs : passed to train_single_lstm (epochs, hidden_size, etc.)

    Returns
    -------
    models : list of nn.Module
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS
    models = []
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"    Seed {seed} ({i+1}/{len(seeds)})...", end=" ")
        m = train_single_lstm(X_tr, y_tr, seed=seed, **kwargs)
        models.append(m)
        if verbose:
            print("done")
    return models


def predict_ensemble(models, X_te):
    """Average sigmoid probabilities across ensemble members.

    Parameters
    ----------
    models : list of nn.Module
    X_te : np.ndarray, shape (N, lookback, n_features)

    Returns
    -------
    probs : np.ndarray, shape (N,) — mean probability across models
    """
    all_probs = np.stack([predict_probs(m, X_te) for m in models])
    return all_probs.mean(axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# 6. DATA LOADING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_processed_data(processed_dir="../data/processed/", target_mat="GR_Fuel_Energy"):
    """Load features, labels, and aligned returns from the processed directory.

    Returns
    -------
    features : pd.DataFrame
    labels : pd.DataFrame
    raw : pd.DataFrame
    X_raw : np.ndarray
    y_raw : np.ndarray
    dates : pd.DatetimeIndex
    feat_cols : list of str
    """
    features = pd.read_csv(processed_dir + "features.csv",
                           index_col=0, parse_dates=True)
    labels   = pd.read_csv(processed_dir + "crisis_labels.csv",
                           index_col=0, parse_dates=True)
    raw      = pd.read_csv(processed_dir + "aligned_returns.csv",
                           index_col=0, parse_dates=True)

    feat_cols = list(features.columns)
    common    = features.index.intersection(labels.dropna().index)
    X_raw     = features.loc[common].values
    y_raw     = labels.loc[common, target_mat].values
    dates     = common

    return features, labels, raw, X_raw, y_raw, dates, feat_cols


def load_selected_features(processed_dir="../data/processed/"):
    """Load SHAP-selected feature names if available.

    Returns
    -------
    list of str or None
    """
    path = processed_dir + "top_shap_features.csv"
    try:
        df = pd.read_csv(path)
        return list(df["feature"].values)
    except FileNotFoundError:
        return None


def prepare_train_test(X_raw, y_raw, dates, lookback, lead,
                       train_ratio=0.75, scaler=None):
    """Scale, sequence, and split data into train/test sets.

    Parameters
    ----------
    X_raw : np.ndarray, shape (T, n_features)
    y_raw : np.ndarray, shape (T,)
    dates : array-like
    lookback, lead : int
    train_ratio : float
    scaler : sklearn scaler or None (creates MinMaxScaler)

    Returns
    -------
    dict with keys: X_tr, X_te, y_tr, y_te, d_te, scaler, n_tr
    """
    from sklearn.preprocessing import MinMaxScaler

    if scaler is None:
        scaler = MinMaxScaler()

    # Fit scaler on training portion only
    n_raw_tr = int(len(X_raw) * train_ratio)
    scaler.fit(X_raw[:n_raw_tr])
    X_sc = scaler.transform(X_raw)

    X_seq, y_seq, d_seq = make_sequences(X_sc, y_raw, lookback, lead,
                                         dates=dates)

    n_tr = int(len(X_seq) * train_ratio)
    return {
        "X_tr":    X_seq[:n_tr],
        "X_te":    X_seq[n_tr:],
        "y_tr":    y_seq[:n_tr],
        "y_te":    y_seq[n_tr:],
        "d_te":    d_seq[n_tr:],
        "scaler":  scaler,
        "n_tr":    n_tr,
        "X_seq":   X_seq,
        "y_seq":   y_seq,
        "d_seq":   d_seq,
    }

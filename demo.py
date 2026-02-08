"""
MNIST Demo — Self-Contained Checkpoint Evaluator
==================================================

Self-contained evaluation script. Downloads MNIST, loads a checkpoint,
computes features, runs ridge regression + t-SNE, and produces a full
report with multiple charts:

  1. t-SNE embedding scatter (colored by digit)
  2. Confusion matrix heatmap
  3. Per-class accuracy bar chart
  4. Neuron activation distribution (per activation function)
  5. Top/bottom neuron heatmaps (highest/lowest Fisher scores)
  6. Feature correlation matrix
  7. Ridge regression sweep curve

All outputs saved to an `output/` subfolder as PNGs + a log file.

Usage: python demo.py
       python demo.py --all   (evaluate ALL checkpoints sequentially)

Requires: torch, torchvision, numpy, scikit-learn, matplotlib
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import time
import json
import gc
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"

PCA_COMPONENTS = 128
SUBSAMPLE_TRAIN = 50000
SUBSAMPLE_TEST = 10000
VAL_FRACTION = 0.2
SEED = 42
ENCODE_BATCH = 500
IMG_PIXELS = 784
IN_CHANNELS = 1

POOL_SHAPES = {
    'avg':  (1, 1),
    '2x2':  (2, 2),
    '3x3':  (3, 3),
    '1x4':  (1, 4),
    '4x1':  (4, 1),
    '1x8':  (1, 8),
    '8x1':  (8, 1),
    '2x4':  (2, 4),
    '4x2':  (4, 2),
}

DIGIT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

LOG_LINES = []


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


# ================================================================
# ACTIVATIONS
# ================================================================

def apply_activation(pre, activation):
    if activation == 'relu':
        return torch.relu(pre)
    elif activation == 'abs':
        return torch.abs(pre)
    elif activation == 'sin':
        return torch.sin(pre)
    elif activation == 'cos':
        return torch.cos(pre)
    elif activation == 'gaussian':
        return torch.exp(-pre * pre)
    elif activation == 'leaky_relu':
        return F.leaky_relu(pre, 0.1)
    elif activation == 'square':
        return pre * pre
    else:
        return torch.tanh(pre)


# ================================================================
# SOLVER
# ================================================================

def solve_regularized(XtX, XtY):
    try:
        L = torch.linalg.cholesky(XtX)
        W = torch.cholesky_solve(XtY, L)
        del L
        return W
    except torch._C._LinAlgError:
        return torch.linalg.solve(XtX, XtY)


def reg_sweep(core_feat, core_y, val_feat, val_y,
              full_feat, full_y, test_feat, test_y):
    D, C = core_feat.shape[1], 10
    device = core_feat.device

    Y_core = torch.zeros(len(core_y), C, device=device)
    Y_core.scatter_(1, core_y.unsqueeze(1), 1.0)

    XtX = core_feat.T @ core_feat
    XtY = core_feat.T @ Y_core
    del Y_core

    regs = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
            1.0, 10.0, 100.0, 1e3, 1e4]
    best_val, best_reg = 0, 1e-3
    results = []

    for reg in regs:
        XtX.diagonal().add_(reg)
        W = solve_regularized(XtX, XtY)
        XtX.diagonal().sub_(reg)
        val_preds = (val_feat @ W).argmax(1)
        val_acc = (val_preds == val_y).float().mean().item()
        tr_preds = (core_feat @ W).argmax(1)
        tr_acc = (tr_preds == core_y).float().mean().item()
        results.append((reg, val_acc, tr_acc))
        if val_acc > best_val:
            best_val = val_acc
            best_reg = reg

    Y_full = torch.zeros(len(val_y), C, device=device)
    Y_full.scatter_(1, val_y.unsqueeze(1), 1.0)
    XtX.addmm_(val_feat.T, val_feat)
    XtY.addmm_(val_feat.T, Y_full)
    del Y_full

    XtX.diagonal().add_(best_reg)
    W = solve_regularized(XtX, XtY)
    del XtX, XtY
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    test_preds = (test_feat @ W).argmax(1)
    test_acc = (test_preds == test_y).float().mean().item()

    del W
    gc.collect()
    return best_val, test_acc, best_reg, test_preds, results


# ================================================================
# V1/V3-L1 CONV ENCODING
# ================================================================

def group_detectors(detectors):
    groups = {}
    for w, b, act, ps, nn in detectors:
        key = (ps, act)
        if key not in groups:
            groups[key] = {'filters': [], 'biases': []}
        groups[key]['filters'].append(w)
        groups[key]['biases'].append(b)
    for key in groups:
        groups[key]['filters'] = torch.cat(groups[key]['filters'], 0)
        groups[key]['biases'] = torch.cat(groups[key]['biases'], 0)
        groups[key]['n_neurons'] = groups[key]['filters'].shape[0]
    return groups


def conv_encode_single_pool(images_chw, groups, pool_name,
                             batch_size=ENCODE_BATCH):
    N = images_chw.shape[0]
    device = images_chw.device
    rows, cols = POOL_SHAPES[pool_name]
    all_feats = []
    for i in range(0, N, batch_size):
        batch = images_chw[i:i + batch_size]
        b = batch.shape[0]
        parts = []
        for (ps, act), grp in sorted(groups.items()):
            filters = grp['filters'].to(device)
            biases = grp['biases'].to(device)
            n_g = grp['n_neurons']
            resp = F.conv2d(batch, filters, bias=biases)
            resp = apply_activation(resp, act)
            pooled = F.adaptive_avg_pool2d(resp, (rows, cols))
            parts.append(pooled.reshape(b, n_g * rows * cols))
            del resp
        all_feats.append(torch.cat(parts, dim=1))
    return torch.cat(all_feats, 0)


def conv_encode_all_pools(images_chw, groups, batch_size=ENCODE_BATCH):
    n_neurons = sum(g['n_neurons'] for g in groups.values())
    pool_names = []
    total = 0
    for name in POOL_SHAPES:
        rows, cols = POOL_SHAPES[name]
        added = n_neurons * rows * cols
        if total + added <= 100000:
            pool_names.append(name)
            total += added
    if not pool_names:
        pool_names = ['avg']
    log(f"  Using pools: {pool_names} ({total} conv dims)")

    N = images_chw.shape[0]
    device = images_chw.device
    pool_specs = [(name, POOL_SHAPES[name]) for name in pool_names]
    all_feats = []
    for i in range(0, N, batch_size):
        batch = images_chw[i:i + batch_size]
        b = batch.shape[0]
        parts = []
        for (ps, act), grp in sorted(groups.items()):
            filters = grp['filters'].to(device)
            biases = grp['biases'].to(device)
            n_g = grp['n_neurons']
            resp = F.conv2d(batch, filters, bias=biases)
            resp = apply_activation(resp, act)
            for pn, (rows, cols) in pool_specs:
                pooled = F.adaptive_avg_pool2d(resp, (rows, cols))
                parts.append(pooled.reshape(b, n_g * rows * cols))
            del resp
        all_feats.append(torch.cat(parts, dim=1))
    return torch.cat(all_feats, 0), pool_names


def get_neuron_activations(images_chw, groups, batch_size=ENCODE_BATCH):
    """Get avg-pool activations per neuron with activation function labels."""
    N = images_chw.shape[0]
    device = images_chw.device
    all_feats = []
    act_labels = []
    for i in range(0, N, batch_size):
        batch = images_chw[i:i + batch_size]
        b = batch.shape[0]
        parts = []
        for (ps, act), grp in sorted(groups.items()):
            filters = grp['filters'].to(device)
            biases = grp['biases'].to(device)
            n_g = grp['n_neurons']
            resp = F.conv2d(batch, filters, bias=biases)
            resp = apply_activation(resp, act)
            pooled = resp.mean(dim=(2, 3))  # avg pool -> (b, n_g)
            parts.append(pooled)
            if i == 0:
                act_labels.extend([act] * n_g)
            del resp
        all_feats.append(torch.cat(parts, dim=1))
    return torch.cat(all_feats, 0), act_labels


# ================================================================
# V2 SPARSE ENCODING
# ================================================================

def sparse_encode(images_flat, detectors, batch_size=ENCODE_BATCH):
    N = images_flat.shape[0]
    device = images_flat.device
    all_feats = []
    for i in range(0, N, batch_size):
        batch = images_flat[i:i + batch_size]
        parts = []
        for indices, weights, bias, activation, nc, nn in detectors:
            idx = indices.to(device)
            w = weights.to(device)
            bi = bias.to(device)
            gathered = batch[:, idx]
            pre = (gathered * w.unsqueeze(0)).sum(-1) + bi.unsqueeze(0)
            parts.append(apply_activation(pre, activation))
            del gathered, pre, idx, w, bi
        all_feats.append(torch.cat(parts, dim=1))
    return torch.cat(all_feats, 0)


def get_sparse_activations(images_flat, detectors, batch_size=ENCODE_BATCH):
    """Get per-neuron activations with activation function labels."""
    N = images_flat.shape[0]
    device = images_flat.device
    all_feats = []
    act_labels = []
    first = True
    for i in range(0, N, batch_size):
        batch = images_flat[i:i + batch_size]
        parts = []
        for indices, weights, bias, activation, nc, nn in detectors:
            idx = indices.to(device)
            w = weights.to(device)
            bi = bias.to(device)
            gathered = batch[:, idx]
            pre = (gathered * w.unsqueeze(0)).sum(-1) + bi.unsqueeze(0)
            parts.append(apply_activation(pre, activation))
            if first:
                act_labels.extend([activation] * nn)
            del gathered, pre, idx, w, bi
        first = False
        all_feats.append(torch.cat(parts, dim=1))
    return torch.cat(all_feats, 0), act_labels


# ================================================================
# DATA LOADING — downloads MNIST automatically
# ================================================================

def load_mnist(device):
    from torchvision import datasets

    log("Loading MNIST (will download if needed)...")
    data_dir = SCRIPT_DIR / "data"
    ds_train = datasets.MNIST(root=str(data_dir), train=True, download=True)
    ds_test = datasets.MNIST(root=str(data_dir), train=False, download=True)

    train_x = ds_train.data.float().reshape(-1, 784) / 255.0
    train_y = ds_train.targets.clone()
    test_x = ds_test.data.float().reshape(-1, 784) / 255.0
    test_y = ds_test.targets.clone()

    torch.manual_seed(SEED)
    if SUBSAMPLE_TRAIN < len(train_y):
        idx = torch.randperm(len(train_y))[:SUBSAMPLE_TRAIN]
        train_x = train_x[idx]
        train_y = train_y[idx]
    if SUBSAMPLE_TEST < len(test_y):
        idx = torch.randperm(len(test_y))[:SUBSAMPLE_TEST]
        test_x = test_x[idx]
        test_y = test_y[idx]

    mu = train_x.mean(0, keepdim=True)
    std = train_x.std(0, keepdim=True).clamp(min=1e-6)
    train_x_norm = (train_x - mu) / std
    test_x_norm = (test_x - mu) / std

    log(f"  Computing PCA ({PCA_COMPONENTS} components)...")
    cov = (train_x_norm.T @ train_x_norm) / (len(train_x_norm) - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    idx_sorted = eigvals.argsort(descending=True)
    eigvals = eigvals[idx_sorted[:PCA_COMPONENTS]]
    eigvecs = eigvecs[:, idx_sorted[:PCA_COMPONENTS]]
    whiten = eigvecs / eigvals.sqrt().unsqueeze(0)

    pca_train = train_x_norm @ whiten
    pca_test = test_x_norm @ whiten

    log(f"  Train: {train_x.shape[0]}, Test: {test_x.shape[0]}")
    log(f"  PCA: {PCA_COMPONENTS} components")

    n_total = len(train_y)
    n_val = int(n_total * VAL_FRACTION)
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))

    return {
        'pca_train': pca_train.to(device),
        'pca_test': pca_test.to(device),
        'train_y': train_y.to(device),
        'test_y': test_y.to(device),
        'train_x_norm': train_x_norm,
        'test_x_norm': test_x_norm,
        'train_chw': train_x_norm.reshape(-1, 28, 28).unsqueeze(1).contiguous().to(device),
        'test_chw': test_x_norm.reshape(-1, 28, 28).unsqueeze(1).contiguous().to(device),
        'perm': perm.to(device),
        'n_val': n_val,
    }


# ================================================================
# CHECKPOINT DISCOVERY
# ================================================================

def find_checkpoints():
    ckpts = []
    for pt in sorted(SCRIPT_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            ckpt = torch.load(pt, map_location='cpu', weights_only=False)
            fmt = ckpt.get('format', 'unknown')
            n = ckpt.get('n_neurons', '?')
            ds = ckpt.get('dataset', '?')
            ckpts.append({
                'path': pt,
                'format': fmt,
                'n_neurons': n,
                'dataset': ds,
                'ckpt': ckpt,
            })
        except Exception as e:
            log(f"  Warning: could not load {pt.name}: {e}")
    return ckpts


# ================================================================
# CHART GENERATORS
# ================================================================

def plot_tsne(embedding, labels_np, ckpt_name, n_neurons, n_features, out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=150)
    for digit in range(10):
        mask = labels_np == digit
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=DIGIT_COLORS[digit], label=str(digit),
                   s=8, alpha=0.6, edgecolors='none')
    ax.legend(title="Digit", fontsize=10, title_fontsize=11,
              markerscale=3, loc='upper right')
    ax.set_title(f"t-SNE Embedding — {ckpt_name}\n"
                 f"{n_neurons}n, {n_features} features, "
                 f"{len(labels_np)} test samples", fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    path = out_dir / "01_tsne.png"
    fig.savefig(str(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def plot_confusion_matrix(test_preds_np, labels_np, ckpt_name, out_dir):
    cm = confusion_matrix(labels_np, test_preds_np, labels=list(range(10)))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(1, 1, figsize=(9, 8), dpi=150)
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
    for i in range(10):
        for j in range(10):
            val = cm_pct[i, j]
            color = 'white' if val > 60 else 'black'
            ax.text(j, i, f"{val:.1f}", ha='center', va='center',
                    fontsize=9, color=color)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix (%) — {ckpt_name}", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8, label='%')
    path = out_dir / "02_confusion_matrix.png"
    fig.savefig(str(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def plot_per_class_accuracy(test_preds_np, labels_np, ckpt_name, out_dir):
    accs = []
    for d in range(10):
        mask = labels_np == d
        acc = (test_preds_np[mask] == d).mean() * 100
        accs.append(acc)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=150)
    bars = ax.bar(range(10), accs, color=DIGIT_COLORS, edgecolor='black',
                  linewidth=0.5)
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(min(accs) - 3, 100.5)
    ax.set_title(f"Per-Class Accuracy — {ckpt_name}\n"
                 f"Overall: {(test_preds_np == labels_np).mean()*100:.2f}%",
                 fontsize=13)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha='center', va='bottom', fontsize=9)
    ax.axhline(y=np.mean(accs), color='gray', linestyle='--', alpha=0.7,
               label=f"Mean: {np.mean(accs):.1f}%")
    ax.legend(fontsize=10)
    path = out_dir / "03_per_class_accuracy.png"
    fig.savefig(str(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def plot_neuron_activations(activations_np, act_labels, ckpt_name, out_dir):
    """Activation distribution per activation function type."""
    unique_acts = sorted(set(act_labels))
    n_acts = len(unique_acts)
    cols = min(4, n_acts)
    rows = (n_acts + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), dpi=150)
    if n_acts == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, act_name in enumerate(unique_acts):
        ax = axes[i]
        mask = [j for j, a in enumerate(act_labels) if a == act_name]
        if not mask:
            continue
        vals = activations_np[:, mask].flatten()
        # Subsample for histogram speed
        if len(vals) > 500000:
            vals = np.random.choice(vals, 500000, replace=False)
        ax.hist(vals, bins=80, color=DIGIT_COLORS[i % len(DIGIT_COLORS)],
                alpha=0.8, edgecolor='none', density=True)
        ax.set_title(f"{act_name} ({len(mask)}n)", fontsize=11)
        ax.set_xlabel("Activation value")
        ax.set_ylabel("Density")
        p5, p95 = np.percentile(vals, [5, 95])
        ax.set_xlim(p5 - 0.5 * abs(p5), p95 + 0.5 * abs(p95))

    for i in range(n_acts, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Neuron Activation Distributions — {ckpt_name}",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    path = out_dir / "04_neuron_activations.png"
    fig.savefig(str(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def plot_neuron_fisher(activations, labels, act_labels, ckpt_name, out_dir):
    """Bar chart of per-neuron Fisher score, colored by activation function."""
    device = activations.device
    N, D = activations.shape
    gm = activations.mean(0)
    sw = torch.zeros(D, device=device)
    sb = torch.zeros(D, device=device)
    for c in range(10):
        mask = labels == c
        nc = mask.sum().float().clamp(min=1)
        if nc < 2:
            continue
        cf = activations[mask]
        cm = cf.mean(0)
        sw += ((cf - cm.unsqueeze(0)) ** 2).sum(0)
        sb += nc * (cm - gm) ** 2
    fisher = (sb / (sw + 1e-8)).cpu().numpy()

    unique_acts = sorted(set(act_labels))
    act_to_color = {a: DIGIT_COLORS[i % len(DIGIT_COLORS)]
                    for i, a in enumerate(unique_acts)}
    colors = [act_to_color[a] for a in act_labels]

    sorted_idx = np.argsort(fisher)[::-1]

    fig, ax = plt.subplots(1, 1, figsize=(14, 5), dpi=150)
    ax.bar(range(len(fisher)), fisher[sorted_idx],
           color=[colors[i] for i in sorted_idx], edgecolor='none', width=1.0)
    ax.set_xlabel("Neuron (sorted by Fisher score)", fontsize=11)
    ax.set_ylabel("Fisher Score", fontsize=11)
    ax.set_title(f"Per-Neuron Fisher Discriminant — {ckpt_name}", fontsize=13)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=act_to_color[a], label=a) for a in unique_acts]
    ax.legend(handles=handles, title="Activation", fontsize=9,
              title_fontsize=10, ncol=min(4, len(unique_acts)))

    path = out_dir / "05_neuron_fisher.png"
    fig.savefig(str(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path, fisher


def plot_reg_sweep(sweep_results, best_reg, ckpt_name, out_dir):
    regs = [r for r, v, t in sweep_results]
    vals = [v * 100 for r, v, t in sweep_results]
    trains = [t * 100 for r, v, t in sweep_results]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=150)
    ax.semilogx(regs, vals, 'o-', color='#1f77b4', label='Validation', linewidth=2)
    ax.semilogx(regs, trains, 's--', color='#ff7f0e', label='Train', linewidth=1.5,
                alpha=0.7)
    ax.axvline(x=best_reg, color='red', linestyle=':', alpha=0.7,
               label=f'Best reg={best_reg:.0e}')
    ax.set_xlabel("Regularization", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Ridge Regression Sweep — {ckpt_name}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    path = out_dir / "06_reg_sweep.png"
    fig.savefig(str(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def plot_feature_correlation(features_np, ckpt_name, out_dir, max_dim=200):
    """Correlation matrix of a subsample of features."""
    D = features_np.shape[1]
    if D > max_dim:
        idx = np.linspace(0, D - 1, max_dim, dtype=int)
        sub = features_np[:, idx]
        label_note = f" (subsampled {max_dim}/{D})"
    else:
        sub = features_np
        label_note = ""

    corr = np.corrcoef(sub.T)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7), dpi=150)
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_title(f"Feature Correlation{label_note} — {ckpt_name}", fontsize=13)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Feature index")
    fig.colorbar(im, ax=ax, shrink=0.8)
    path = out_dir / "07_feature_correlation.png"
    fig.savefig(str(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def plot_summary_dashboard(ckpt_name, n_neurons, n_features, test_acc,
                           per_class_accs, ckpt_meta, out_dir):
    """Single-page summary dashboard."""
    fig = plt.figure(figsize=(14, 8), dpi=150)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

    # Title area
    fig.suptitle(f"MNIST Demo Report — {ckpt_name}", fontsize=16, y=0.98)

    # Panel 1: Key metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    fmt = ckpt_meta.get('format', '?')
    fmt_label = {"v29": "V1/V3-L1 Conv", "v2_sparse": "V2 Sparse",
                 "v3_l2": "V3 Two-Layer"}.get(fmt, fmt)
    lines = [
        f"Model: {fmt_label}",
        f"Neurons: {n_neurons}",
        f"Features: {n_features}",
        f"Test Acc: {test_acc*100:.2f}%",
        f"Test Samples: {SUBSAMPLE_TEST}",
    ]
    if 'activations' in ckpt_meta:
        lines.append(f"Activations: {len(ckpt_meta['activations'])}")
    ax1.text(0.1, 0.95, "\n".join(lines), transform=ax1.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax1.set_title("Key Metrics", fontsize=12, fontweight='bold')

    # Panel 2: Per-class accuracy bars
    ax2 = fig.add_subplot(gs[0, 1:])
    bars = ax2.bar(range(10), per_class_accs, color=DIGIT_COLORS,
                   edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(10))
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(min(per_class_accs) - 3, 100.5)
    ax2.set_title("Per-Class Accuracy", fontsize=12, fontweight='bold')
    for bar, acc in zip(bars, per_class_accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{acc:.1f}", ha='center', va='bottom', fontsize=8)

    # Panel 3: Activation function distribution (pie)
    ax3 = fig.add_subplot(gs[1, 0])
    if 'activations' in ckpt_meta:
        act_list = ckpt_meta['activations']
        if fmt == 'v29':
            act_counts = {}
            for w, b, act, ps, nn in ckpt_meta.get('detectors', []):
                act_counts[act] = act_counts.get(act, 0) + nn
        elif fmt == 'v2_sparse':
            act_counts = {}
            for idx, w, b, act, nc, nn in ckpt_meta.get('detectors', []):
                act_counts[act] = act_counts.get(act, 0) + nn
        elif fmt == 'v3_l2':
            act_counts = {}
            for idx, w, b, act, nc, nn in ckpt_meta.get('detectors', []):
                act_counts[act] = act_counts.get(act, 0) + nn
        else:
            act_counts = {a: 1 for a in act_list}

        if act_counts:
            labels_pie = list(act_counts.keys())
            sizes = list(act_counts.values())
            colors_pie = [DIGIT_COLORS[i % len(DIGIT_COLORS)]
                         for i in range(len(labels_pie))]
            ax3.pie(sizes, labels=labels_pie, colors=colors_pie,
                    autopct='%1.0f%%', startangle=90, textprops={'fontsize': 9})
        else:
            ax3.text(0.5, 0.5, "N/A", ha='center', va='center')
    else:
        ax3.text(0.5, 0.5, "N/A", ha='center', va='center')
    ax3.set_title("Activation Mix", fontsize=12, fontweight='bold')

    # Panel 4-5: Misclassified examples note
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.axis('off')
    worst = sorted(range(10), key=lambda d: per_class_accs[d])[:3]
    best_d = sorted(range(10), key=lambda d: per_class_accs[d])[-3:]
    summary_lines = [
        f"Overall Test Accuracy: {test_acc*100:.2f}%",
        f"",
        f"Best digits:  {', '.join(f'{d}({per_class_accs[d]:.1f}%)' for d in reversed(best_d))}",
        f"Worst digits: {', '.join(f'{d}({per_class_accs[d]:.1f}%)' for d in worst)}",
        f"Accuracy range: {min(per_class_accs):.1f}% — {max(per_class_accs):.1f}%",
        f"Std dev: {np.std(per_class_accs):.2f}%",
    ]
    ax4.text(0.1, 0.85, "\n".join(summary_lines), transform=ax4.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    ax4.set_title("Summary", fontsize=12, fontweight='bold')

    path = out_dir / "00_summary_dashboard.png"
    fig.savefig(str(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


# ================================================================
# EVALUATE ONE CHECKPOINT
# ================================================================

def evaluate_checkpoint(selected, ckpts, data, device):
    global LOG_LINES
    LOG_LINES = []

    fmt = selected['format']
    detectors = selected['ckpt']['detectors']
    n_neurons = selected['ckpt']['n_neurons']
    ckpt_name = selected['path'].stem

    fmt_label = {"v29": "V1/V3-L1 Conv", "v2_sparse": "V2 Sparse",
                 "v3_l2": "V3 Two-Layer"}.get(fmt, fmt)
    log(f"{'='*60}")
    log(f"Evaluating: {selected['path'].name}")
    log(f"  Type: {fmt_label}, Neurons: {n_neurons}")
    log(f"  Device: {device}")
    log(f"{'='*60}")

    out_dir = OUTPUT_DIR / ckpt_name
    out_dir.mkdir(parents=True, exist_ok=True)

    test_y = data['test_y']
    train_y = data['train_y']
    labels_np = test_y.cpu().numpy()
    perm = data['perm']
    n_val = data['n_val']
    full_ty = train_y
    core_y = full_ty[perm[n_val:]]
    val_y = full_ty[perm[:n_val]]

    # --- Encode features ---
    log("Encoding features...")
    t0 = time.time()

    activations_tensor = None
    act_labels = None

    with torch.no_grad():
        if fmt == 'v29':
            groups = group_detectors(detectors)
            conv_feats, pools_used = conv_encode_all_pools(data['test_chw'], groups)
            features_test = torch.cat([data['pca_test'], conv_feats], dim=1)
            conv_feats_train, _ = conv_encode_all_pools(data['train_chw'], groups)
            features_train = torch.cat([data['pca_train'], conv_feats_train], dim=1)
            # Neuron activations (avg pool only for analysis)
            activations_tensor, act_labels = get_neuron_activations(
                data['test_chw'], groups)
            del conv_feats, conv_feats_train

        elif fmt == 'v2_sparse':
            test_flat = data['test_x_norm'].to(device)
            train_flat = data['train_x_norm'].to(device)
            sparse_test = sparse_encode(test_flat, detectors)
            sparse_train = sparse_encode(train_flat, detectors)
            features_test = torch.cat([data['pca_test'], sparse_test], dim=1)
            features_train = torch.cat([data['pca_train'], sparse_train], dim=1)
            activations_tensor, act_labels = get_sparse_activations(
                test_flat, detectors)
            del sparse_test, sparse_train, test_flat, train_flat

        elif fmt == 'v3_l2':
            l1_neurons = selected['ckpt'].get('l1_neurons', 300)
            log(f"  V3-L2 needs L1 ({l1_neurons}n)...")
            l1_ckpt = None
            for c2 in ckpts:
                if c2['format'] == 'v29' and c2['n_neurons'] == l1_neurons:
                    l1_ckpt = c2
                    break
            if l1_ckpt is None:
                log(f"  ERROR: No L1 checkpoint with {l1_neurons}n.")
                return
            log(f"  Found L1: {l1_ckpt['path'].name}")
            l1_groups = group_detectors(l1_ckpt['ckpt']['detectors'])

            l1_avg_test = conv_encode_single_pool(data['test_chw'], l1_groups, 'avg')
            l1_avg_train = conv_encode_single_pool(data['train_chw'], l1_groups, 'avg')
            l1_mean = l1_avg_train.mean(0, keepdim=True)
            l1_std = l1_avg_train.std(0, keepdim=True).clamp(min=1e-8)
            l1_avg_test_z = (l1_avg_test - l1_mean) / l1_std
            l1_avg_train_z = (l1_avg_train - l1_mean) / l1_std

            l2_test = sparse_encode(l1_avg_test_z, detectors)
            l2_train = sparse_encode(l1_avg_train_z, detectors)

            conv_test, _ = conv_encode_all_pools(data['test_chw'], l1_groups)
            conv_train, _ = conv_encode_all_pools(data['train_chw'], l1_groups)

            features_test = torch.cat([data['pca_test'], conv_test, l2_test], dim=1)
            features_train = torch.cat([data['pca_train'], conv_train, l2_train], dim=1)

            activations_tensor, act_labels = get_sparse_activations(
                l1_avg_test_z, detectors)
            del (l1_avg_test, l1_avg_train, l1_avg_test_z, l1_avg_train_z,
                 l2_test, l2_train, conv_test, conv_train, l1_mean, l1_std)
        else:
            log(f"  Unknown format: {fmt}")
            return

    n_features = features_test.shape[1]
    log(f"  Features: {n_features} dims [{time.time()-t0:.1f}s]")

    # --- Z-score ---
    features_train_p = features_train[perm]
    feat_mean = features_train_p.mean(0, keepdim=True)
    feat_std = features_train_p.std(0, keepdim=True).clamp(min=1e-8)
    features_train_p = (features_train_p - feat_mean) / feat_std
    features_test_z = (features_test - feat_mean) / feat_std
    del feat_mean, feat_std

    # --- Ridge regression ---
    log("Running ridge regression sweep...")
    val_feat = features_train_p[:n_val]
    core_feat = features_train_p[n_val:]

    best_val, best_test, best_reg, test_preds, sweep = reg_sweep(
        core_feat, core_y, val_feat, val_y,
        features_train_p, full_ty[perm], features_test_z, test_y)

    test_preds_np = test_preds.cpu().numpy()

    per_class_accs = []
    for d in range(10):
        mask = labels_np == d
        per_class_accs.append((test_preds_np[mask] == d).mean() * 100)

    log(f"  Val:  {best_val*100:.2f}%")
    log(f"  Test: {best_test*100:.2f}%")
    log(f"  Reg:  {best_reg:.0e}")
    log(f"  Per-class: {' '.join(f'{d}:{a:.1f}%' for d, a in enumerate(per_class_accs))}")

    # --- Generate charts ---
    log("Generating charts...")

    # 0. Summary dashboard
    p = plot_summary_dashboard(ckpt_name, n_neurons, n_features, best_test,
                               per_class_accs, selected['ckpt'], out_dir)
    log(f"  Saved: {p.name}")

    # 1. t-SNE
    log("  Running t-SNE...")
    t0 = time.time()
    feat_np = features_test_z.cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto',
                init='pca', random_state=SEED, n_jobs=-1)
    embedding = tsne.fit_transform(feat_np)
    log(f"  t-SNE done [{time.time()-t0:.1f}s]")
    p = plot_tsne(embedding, labels_np, ckpt_name, n_neurons, n_features, out_dir)
    log(f"  Saved: {p.name}")

    # 2. Confusion matrix
    p = plot_confusion_matrix(test_preds_np, labels_np, ckpt_name, out_dir)
    log(f"  Saved: {p.name}")

    # 3. Per-class accuracy
    p = plot_per_class_accuracy(test_preds_np, labels_np, ckpt_name, out_dir)
    log(f"  Saved: {p.name}")

    # 4. Neuron activation distributions
    if activations_tensor is not None and act_labels:
        act_np = activations_tensor.cpu().numpy()
        p = plot_neuron_activations(act_np, act_labels, ckpt_name, out_dir)
        log(f"  Saved: {p.name}")

        # 5. Fisher scores
        p, fisher = plot_neuron_fisher(
            activations_tensor, test_y, act_labels, ckpt_name, out_dir)
        log(f"  Saved: {p.name}")
        log(f"    Fisher — max: {fisher.max():.2f}, "
            f"median: {np.median(fisher):.2f}, "
            f"min: {fisher.min():.4f}")

    # 6. Reg sweep curve
    p = plot_reg_sweep(sweep, best_reg, ckpt_name, out_dir)
    log(f"  Saved: {p.name}")

    # 7. Feature correlation
    p = plot_feature_correlation(feat_np, ckpt_name, out_dir)
    log(f"  Saved: {p.name}")

    # --- Save log + JSON report ---
    report = {
        'checkpoint': selected['path'].name,
        'format': fmt,
        'format_label': fmt_label,
        'n_neurons': n_neurons,
        'n_features': n_features,
        'test_accuracy': best_test,
        'val_accuracy': best_val,
        'best_reg': best_reg,
        'per_class_accuracy': {str(d): a for d, a in enumerate(per_class_accs)},
        'reg_sweep': [{'reg': r, 'val': v, 'train': t} for r, v, t in sweep],
        'test_samples': SUBSAMPLE_TEST,
        'train_samples': SUBSAMPLE_TRAIN,
        'pca_components': PCA_COMPONENTS,
    }
    with open(out_dir / "report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)

    with open(out_dir / "log.txt", 'w') as f:
        f.write("\n".join(LOG_LINES) + "\n")

    log(f"\nAll outputs saved to: {out_dir}")
    log(f"  Test accuracy: {best_test*100:.2f}%")

    # Cleanup
    del features_train, features_test, features_train_p, features_test_z
    del activations_tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return report


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="MNIST Demo — Checkpoint Evaluator")
    parser.add_argument('--all', action='store_true',
                        help='Evaluate ALL checkpoints sequentially')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- Find checkpoints ---
    ckpts = find_checkpoints()
    if not ckpts:
        print(f"No .pt checkpoint files found in {SCRIPT_DIR}")
        print("Copy checkpoint files here or run V1/V2/V3 first.")
        sys.exit(1)

    print(f"\nFound {len(ckpts)} checkpoint(s):\n")
    for i, c in enumerate(ckpts):
        fmt_label = {"v29": "V1/V3-L1 Conv", "v2_sparse": "V2 Sparse",
                     "v3_l2": "V3 Two-Layer"}.get(c['format'], c['format'])
        print(f"  [{i+1}] {c['path'].name}  ({fmt_label}, {c['n_neurons']}n)")

    if args.all:
        selected_list = ckpts
        print(f"\n--all flag: evaluating all {len(ckpts)} checkpoints.\n")
    else:
        print()
        choice = input(f"Choose checkpoint [1-{len(ckpts)}], or 'all': ").strip()
        if choice.lower() == 'all':
            selected_list = ckpts
        else:
            try:
                idx = int(choice) - 1
                assert 0 <= idx < len(ckpts)
                selected_list = [ckpts[idx]]
            except (ValueError, AssertionError):
                print("Invalid choice.")
                sys.exit(1)

    # --- Load data once ---
    data = load_mnist(device)

    # --- Evaluate ---
    all_reports = []
    for sel in selected_list:
        report = evaluate_checkpoint(sel, ckpts, data, device)
        if report:
            all_reports.append(report)

    # --- Comparison table if multiple ---
    if len(all_reports) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON ACROSS ALL CHECKPOINTS")
        print(f"{'='*70}")
        hdr = f"{'Checkpoint':<45} {'Type':<16} {'Neurons':>7} {'Features':>8} {'Test%':>7}"
        print(hdr)
        print("-" * len(hdr))
        for r in sorted(all_reports, key=lambda x: x['test_accuracy'], reverse=True):
            print(f"{r['checkpoint']:<45} {r['format_label']:<16} "
                  f"{r['n_neurons']:>7} {r['n_features']:>8} "
                  f"{r['test_accuracy']*100:>6.2f}%")

        # Save comparison
        comp_path = OUTPUT_DIR / "comparison.json"
        with open(comp_path, 'w') as f:
            json.dump(all_reports, f, indent=2, default=str)
        print(f"\nComparison saved to: {comp_path}")

    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

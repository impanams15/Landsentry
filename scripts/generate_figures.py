"""
Publication-quality figure generator for LandSentry experiments.
Reads from outputs/final_experiment_summary.json — does NOT retrain or modify anything.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import NpyLandslideDataset
from models.siamese_cnn import SiameseChangeDetector
from models.transformer_cd import TransformerChangeDetector

# ── Helpers ─────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DARK   = '#1C1C2E'
BLUE   = '#4C72B0'
ORG    = '#DD8452'
EXPERIMENTS = [
    "Kodagu -> Kodagu",
    "Kodagu -> Wayanad",
    "Wayanad -> Wayanad",
    "Wayanad -> Kodagu",
]
COLORS = {'Siamese CNN': BLUE, 'Transformer': ORG}

def load_summary():
    with open('outputs/final_experiment_summary.json') as f:
        return json.load(f)

def parse(summary):
    res = {'Siamese CNN': {}, 'Transformer': {}}
    for exp in summary:
        res[exp['model']][exp['dataset']] = {
            'dice': exp['Dice'], 'iou': exp['IoU'],
            'precision': exp['Precision'], 'recall': exp['Recall'],
        }
    return res

def ensure(d):
    os.makedirs(d, exist_ok=True)

def load_ckpt(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    in_ch = ckpt['in_channels']
    name  = ckpt['model_name']
    if name == 'siamese':
        m = SiameseChangeDetector(in_channels=in_ch)
    else:
        m = TransformerChangeDetector(in_channels=in_ch)
    m.load_state_dict(ckpt['model_state'])
    m.eval()
    return m.to(DEVICE), in_ch

def to_rgb(t):
    """Convert [C, H, W] tensor to display RGB using first 3 channels."""
    arr = t[:3].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)

def predict(model, pre, post, in_ch):
    """Run model inference. Slices tensors to match checkpoint in_channels."""
    pre_b  = pre.unsqueeze(0).to(DEVICE)[:, :in_ch]
    post_b = post.unsqueeze(0).to(DEVICE)[:, :in_ch]
    with torch.no_grad():
        return torch.sigmoid(model(pre_b, post_b))[0, 0].cpu().numpy()

def pick_interesting_sample(ds, prefer_positive=True):
    """Return (pre, post, mask) for the first sample with positive pixels, fallback index 0."""
    for i in range(len(ds)):
        pre, post, mask = ds[i]
        if prefer_positive and mask.sum() > 0:
            return pre, post, mask
    return ds[0]

# ── Figure 1: Dice + IoU side-by-side bars ──────────────────────────────────
def fig1(metrics, out):
    short = ["K→K", "K→W", "W→W", "W→K"]
    x = np.arange(len(EXPERIMENTS))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in zip(axes, ['dice', 'iou'], ['Dice Coefficient', 'IoU']):
        for offset, model in zip([-w/2, w/2], ['Siamese CNN', 'Transformer']):
            vals = [metrics[model][e][metric] for e in EXPERIMENTS]
            bars = ax.bar(x + offset, vals, w, label=model,
                          color=COLORS[model], alpha=0.87, zorder=3)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(short, fontsize=10)
        ax.set_ylim(0, 1.05); ax.set_ylabel(metric.upper(), fontsize=10)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
        ax.legend(fontsize=9)
    fig.suptitle('Cross-Event Dice and IoU Comparison — LandSentry', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'cross_event_dice_iou.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] cross_event_dice_iou.png")

# ── Figure 2: Precision vs Recall scatter ───────────────────────────────────
def fig2(metrics, out):
    mkrs = ['o', 's', '^', 'D']
    fig, ax = plt.subplots(figsize=(8, 7))
    for exp, mkr in zip(EXPERIMENTS, mkrs):
        short = exp.replace(' ', '').replace('->', '→')
        for model in ['Siamese CNN', 'Transformer']:
            p = metrics[model][exp]['precision']
            r = metrics[model][exp]['recall']
            ax.scatter(r, p, marker=mkr, color=COLORS[model], s=120, zorder=5,
                       label=f'{model[:7]} {short}')
            ax.annotate(f'{short}', (r, p), textcoords='offset points',
                        xytext=(5, 3), fontsize=7, color=COLORS[model])
    ax.set_xlabel('Recall', fontsize=11); ax.set_ylabel('Precision', fontsize=11)
    ax.set_xlim(-0.05, 1.1); ax.set_ylim(-0.05, 1.1)
    ax.set_title('Precision vs Recall — All Experiment Conditions', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    # Deduplicated legend
    handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=BLUE, markersize=10, label='Siamese CNN'),
               plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=ORG,  markersize=10, label='Transformer')]
    ax.legend(handles=handles, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'precision_recall_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] precision_recall_comparison.png")

# ── Figure 3: 2×2 Generalization matrix heatmaps ───────────────────────────
def heatmap(model_name, metrics, out):
    data = np.array([
        [metrics[model_name]['Kodagu -> Kodagu']['dice'], metrics[model_name]['Kodagu -> Wayanad']['dice']],
        [metrics[model_name]['Wayanad -> Kodagu']['dice'], metrics[model_name]['Wayanad -> Wayanad']['dice']],
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Test: Kodagu', 'Test: Wayanad'], fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Train: Kodagu', 'Train: Wayanad'], fontsize=11)
    for i in range(2):
        for j in range(2):
            v = data[i, j]
            col = 'white' if v < 0.4 else 'black'
            ax.text(j, i, f'{v:.4f}', ha='center', va='center', fontsize=14, color=col, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Dice')
    ax.set_title(f'{model_name}\nCross-Event Generalization Matrix (Dice)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname = f"{model_name.lower().replace(' ', '_')}_generalization_matrix.png"
    plt.savefig(os.path.join(out, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {fname}")

def fig3(metrics, out):
    heatmap('Siamese CNN', metrics, out)
    heatmap('Transformer', metrics, out)

# ── Figure 4: Qualitative segmentation panels ───────────────────────────────
def qualitative(train_src, test_src, s_ckpt_path, t_ckpt_path, data_root, out):
    s_model, s_in = load_ckpt(s_ckpt_path)
    t_model, t_in = load_ckpt(t_ckpt_path)
    ds = NpyLandslideDataset(data_root)
    pre, post, mask = pick_interesting_sample(ds)

    s_prob = predict(s_model, pre, post, s_in)
    t_prob = predict(t_model, pre, post, t_in)
    s_pred = (s_prob > 0.5).astype(float)
    t_pred = (t_prob > 0.5).astype(float)

    pre_rgb  = to_rgb(pre)
    post_rgb = to_rgb(post)
    gt       = mask.squeeze().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    panels = [(pre_rgb,  'Pre-event RGB',     'viridis', None, None),
              (post_rgb, 'Post-event RGB',    'viridis', None, None),
              (gt,       'Ground Truth',       'Greys_r', 0, 1),
              (s_pred,   'Siamese Pred\n(thr=0.5)', 'Greys_r', 0, 1),
              (t_pred,   'Transformer Pred\n(thr=0.5)', 'Greys_r', 0, 1)]
    for ax, (img, title, cmap, vmin, vmax) in zip(axes, panels):
        kw = dict(cmap=cmap)
        if vmin is not None: kw.update(vmin=vmin, vmax=vmax)
        ax.imshow(img, **kw)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
    fig.suptitle(f'Qualitative Results  —  Train: {train_src}  →  Test: {test_src}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = f"qualitative_{train_src.lower()}_to_{test_src.lower()}.png"
    plt.savefig(os.path.join(out, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {fname}")

def fig4(out):
    configs = [
        ("Kodagu",  "Kodagu",  "checkpoints/siamese_kodagu_best.pt",  "checkpoints/transformer_kodagu_best.pt",  "data/ml/kodagu/test"),
        ("Kodagu",  "Wayanad", "checkpoints/siamese_kodagu_best.pt",  "checkpoints/transformer_kodagu_best.pt",  "data/ml/wayanad/test"),
        ("Wayanad", "Wayanad", "checkpoints/siamese_wayanad_best.pt", "checkpoints/transformer_wayanad_best.pt", "data/ml/wayanad/test"),
        ("Wayanad", "Kodagu",  "checkpoints/siamese_wayanad_best.pt", "checkpoints/transformer_wayanad_best.pt", "data/ml/kodagu/test"),
    ]
    for train_src, test_src, s_ckpt, t_ckpt, data_root in configs:
        qualitative(train_src, test_src, s_ckpt, t_ckpt, data_root, out)

# ── Figure 5: Wayanad Transformer metric profile ────────────────────────────
def fig5(metrics, out):
    d = metrics['Transformer']['Wayanad -> Wayanad']
    labels = ['Dice', 'IoU', 'Precision', 'Recall']
    values = [d['dice'], d['iou'], d['precision'], d['recall']]
    pal = ['#4C72B0', '#55A868', '#DD8452', '#C44E52']
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=pal, width=0.55, zorder=3)
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_title('Metric Profile: Transformer  —  Wayanad → Wayanad',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f'{v:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'wayanad_transformer_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] wayanad_transformer_metrics.png")

# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    out_dir = 'outputs/figures'
    ensure(out_dir)
    summary = load_summary()
    metrics = parse(summary)

    print("Generating figures...")
    fig1(metrics, out_dir)
    fig2(metrics, out_dir)
    fig3(metrics, out_dir)
    fig4(out_dir)
    # fig5(metrics, out_dir)  # Removed single-experiment metric profile figure
    print("\nAll figures saved to outputs/figures/")

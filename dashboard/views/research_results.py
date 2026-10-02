def render_page():
    """
    LandSentry Results Dashboard — Research Experiment Visualization
    Runs independently of dashboard/app.py (GEE image-fetching dashboard).

    Launch: python -m streamlit run dashboard/results_app.py

    READ-ONLY: does NOT retrain models, modify checkpoints, or touch datasets.
    """
    import os, sys, json
    import numpy as np
    import streamlit as st
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ── path setup so we can import project modules ──────────────────────────────
    # ── path setup so we can import project modules ──────────────────────────────
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    ROOT = str(PROJECT_ROOT)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    from data.dataset import NpyLandslideDataset
    from models.siamese_cnn import SiameseChangeDetector
    from models.transformer_cd import TransformerChangeDetector

    # ── constants ────────────────────────────────────────────────────────────────
    SUMMARY_PATH = PROJECT_ROOT / 'outputs' / 'final_experiment_summary.json'
    FIGURES_DIR  = PROJECT_ROOT / 'outputs' / 'figures'
    EXPERIMENTS  = [
        "Kodagu -> Kodagu",
        "Kodagu -> Wayanad",
        "Wayanad -> Wayanad",
        "Wayanad -> Kodagu",
    ]
    CKPT_MAP = {
        "Kodagu":  {
            "Siamese CNN":  str(PROJECT_ROOT / 'checkpoints' / 'siamese_kodagu_best.pt'),
            "Transformer":  str(PROJECT_ROOT / 'checkpoints' / 'transformer_kodagu_best.pt'),
        },
        "Wayanad": {
            "Siamese CNN":  str(PROJECT_ROOT / 'checkpoints' / 'siamese_wayanad_best.pt'),
            "Transformer":  str(PROJECT_ROOT / 'checkpoints' / 'transformer_wayanad_best.pt'),
        },
    }
    DATA_MAP = {
        "Kodagu":  str(PROJECT_ROOT / 'data' / 'ml' / 'kodagu' / 'test'),
        "Wayanad": str(PROJECT_ROOT / 'data' / 'ml' / 'wayanad' / 'test'),
    }
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── loaders (cached) ─────────────────────────────────────────────────────────
    @st.cache_data
    def load_summary():
        if not SUMMARY_PATH.exists():
            raise FileNotFoundError(
                f"LandSentry experiment summary not found at: {SUMMARY_PATH}"
            )
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    @st.cache_data
    def parse_metrics(summary):
        res = {}
        for exp in summary:
            key = (exp['model'], exp['dataset'])
            res[key] = exp
        return res

    @st.cache_resource
    def load_model(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        name, in_ch = ckpt['model_name'], ckpt['in_channels']
        if name == 'siamese':
            m = SiameseChangeDetector(in_channels=in_ch)
        else:
            m = TransformerChangeDetector(in_channels=in_ch)
        m.load_state_dict(ckpt['model_state'])
        m.eval()
        return m.to(DEVICE), in_ch, ckpt

    @st.cache_data
    def load_dataset(root):
        ds = NpyLandslideDataset(root)
        return ds

    def to_rgb(tensor):
        arr = tensor[:3].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)

    def run_inference(model, in_ch, pre, post):
        pre_b  = pre.unsqueeze(0).to(DEVICE)[:, :in_ch]
        post_b = post.unsqueeze(0).to(DEVICE)[:, :in_ch]
        with torch.no_grad():
            prob = torch.sigmoid(model(pre_b, post_b))[0, 0].cpu().numpy()
        return prob

    # ── page config ───────────────────────────────────────────────────────────────

    # ── CSS ───────────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1e2a3a 0%, #0f1b2d 100%);
        border: 1px solid #2d4a6a;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        color: #e8f4fd;
    }
    .metric-card .label { font-size: 13px; color: #8ab4d4; font-weight: 500; margin-bottom: 4px; }
    .metric-card .value { font-size: 28px; font-weight: 700; color: #4fc3f7; }
    .section-header {
        font-size: 22px; font-weight: 700;
        color: #4fc3f7; margin-top: 8px; margin-bottom: 4px;
    }
    .exp-tag {
        display: inline-block;
        background: #1a3a5c; color: #7ec8e3;
        border-radius: 8px; padding: 4px 10px;
        font-size: 12px; margin: 2px;
    }
    .warning-box {
        background: #2b1a00; border: 1px solid #c96000;
        border-radius: 8px; padding: 12px 16px;
        color: #ffb85f; font-size: 13px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── load data ─────────────────────────────────────────────────────────────────
    summary  = load_summary()
    metrics  = parse_metrics(summary)

    # ── sidebar ───────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/64/satellite.png", width=60)
        st.title("LandSentry")
        st.caption("Research Results Dashboard")
        st.divider()
        tab_choice = st.radio("Navigate", [
            "📋 Overview",
            "📊 Model Comparison",
            "🗺️ Cross-Event Generalization",
            "🖼️ Qualitative Results",
            "📑 Metrics Table",
            "⚙️ Experiment Details",
            "📦 Dataset Statistics",
            "🔳 Confusion Matrix",
            "📉 Baseline Comparison",
        ])
        st.divider()
        st.caption("READ-ONLY — no models retrained")

    # ═══════════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════════════
    if tab_choice == "📋 Overview":
        st.markdown("# 🛰️ LandSentry — Experiment Overview")
        st.markdown("**Bitemporal Siamese and Transformer-based Landslide Change Detection · Western Ghats · Cross-Event Generalization**")
        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        for col, label, value in zip(
            [c1, c2, c3, c4],
            ["Models", "Datasets", "Experimental Conditions", "Architecture Types"],
            [2, 2, 4, "CNN + Transformer"],
        ):
            col.markdown(f"""<div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{value}</div></div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("### Experiment Matrix")
        cols = st.columns(4)
        for col, exp in zip(cols, EXPERIMENTS):
            train, test = exp.split(' -> ')
            with col:
                st.markdown(f"**{exp}**")
                s = metrics[("Siamese CNN", exp)]
                t = metrics[("Transformer", exp)]
                st.metric("Siamese Dice",      f"{s['Dice']:.4f}")
                st.metric("Transformer Dice",  f"{t['Dice']:.4f}")

        st.divider()
        st.markdown("### Generated Figures")
        fig_files = [f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')]
        fig_files.sort()
        n_cols = 3
        rows = [fig_files[i:i+n_cols] for i in range(0, len(fig_files), n_cols)]
        for row in rows:
            cols = st.columns(len(row))
            for col, fname in zip(cols, row):
                col.image(os.path.join(FIGURES_DIR, fname), caption=fname.replace('_',' ').replace('.png',''), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    # TAB 2 — MODEL COMPARISON
    # ═══════════════════════════════════════════════════════════════════════════════
    elif tab_choice == "📊 Model Comparison":
        st.markdown("# 📊 Model Comparison")
        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            model_sel = st.selectbox("Model", ["Both", "Siamese CNN", "Transformer"])
        with c2:
            st.markdown("")

        c3, c4 = st.columns(2)
        with c3:
            train_sel = st.selectbox("Training Dataset", ["All", "Kodagu", "Wayanad"])
        with c4:
            test_sel  = st.selectbox("Testing Dataset",  ["All", "Kodagu", "Wayanad"])

        def matches(exp_str, train_f, test_f):
            t, te = exp_str.split(' -> ')
            if train_f != "All" and t != train_f: return False
            if test_f  != "All" and te != test_f: return False
            return True

        model_list = ["Siamese CNN", "Transformer"] if model_sel == "Both" else [model_sel]
        filtered = [e for e in EXPERIMENTS if matches(e, train_sel, test_sel)]

        if not filtered:
            st.warning("No experiments match the selected filters.")
        else:
            for exp in filtered:
                st.markdown(f"### `{exp}`")
                for metric in ["Dice", "IoU", "Precision", "Recall"]:
                    cols = st.columns(len(model_list) + 1)
                    cols[0].markdown(f"**{metric}**")
                    for col, model in zip(cols[1:], model_list):
                        val = metrics[(model, exp)][metric]
                        col.metric(model, f"{val:.4f}")
                st.divider()

            st.markdown("### 📈 Threshold Sensitivity Analysis (F1 vs Decision Threshold)")
            st.caption("Evaluates model performance across decision thresholds from 0.1 to 0.9 for the selected experiment.")

            sens_exp = st.selectbox("Select Experiment for Sensitivity Analysis", EXPERIMENTS, key="sens_exp")

            if st.button("▶ Compute Threshold Sensitivity Curve", type="primary"):
                try:
                    train_k, test_k = sens_exp.split(" -> ")
                    ds_sens = load_dataset(DATA_MAP[test_k])
                    thresholds = np.linspace(0.1, 0.9, 9)
                    f1_s_list, f1_t_list = [], []

                    m_sens_s, in_ch_s, _ = load_model(CKPT_MAP[train_k]["Siamese CNN"])
                    m_sens_t, in_ch_t, _ = load_model(CKPT_MAP[train_k]["Transformer"])

                    for thr in thresholds:
                        # Siamese
                        tp_s = fp_s = fn_s = 0
                        for i in range(len(ds_sens)):
                            pre_s, post_s, mask_s = ds_sens[i]
                            prob = run_inference(m_sens_s, in_ch_s, pre_s, post_s)
                            pred = (prob > thr).astype(int)
                            gt   = (mask_s.squeeze().numpy() > 0.5).astype(int)
                            tp_s += int(((pred == 1) & (gt == 1)).sum())
                            fp_s += int(((pred == 1) & (gt == 0)).sum())
                            fn_s += int(((pred == 0) & (gt == 1)).sum())
                        prec_s = tp_s / (tp_s + fp_s + 1e-8)
                        rec_s  = tp_s / (tp_s + fn_s + 1e-8)
                        f1_s   = 2 * prec_s * rec_s / (prec_s + rec_s + 1e-8)
                        f1_s_list.append(f1_s)

                        # Transformer
                        tp_t = fp_t = fn_t = 0
                        for i in range(len(ds_sens)):
                            pre_t, post_t, mask_t = ds_sens[i]
                            prob = run_inference(m_sens_t, in_ch_t, pre_t, post_t)
                            pred = (prob > thr).astype(int)
                            gt   = (mask_t.squeeze().numpy() > 0.5).astype(int)
                            tp_t += int(((pred == 1) & (gt == 1)).sum())
                            fp_t += int(((pred == 1) & (gt == 0)).sum())
                            fn_t += int(((pred == 0) & (gt == 1)).sum())
                        prec_t = tp_t / (tp_t + fp_t + 1e-8)
                        rec_t  = tp_t / (tp_t + fn_t + 1e-8)
                        f1_t   = 2 * prec_t * rec_t / (prec_t + rec_t + 1e-8)
                        f1_t_list.append(f1_t)

                    fig_sens, ax_sens = plt.subplots(figsize=(7, 4))
                    fig_sens.patch.set_facecolor('#0f1b2d')
                    ax_sens.set_facecolor('#0f1b2d')
                    ax_sens.plot(thresholds, f1_s_list, 'o-', color='#4fc3f7', linewidth=2, label='Siamese CNN')
                    ax_sens.plot(thresholds, f1_t_list, 's-', color='#81c784', linewidth=2, label='Transformer')
                    ax_sens.axvline(x=0.5, color='#e57373', linestyle='--', label='Selected Threshold (0.5)')
                    ax_sens.set_xlabel('Decision Threshold', color='white')
                    ax_sens.set_ylabel('F1 Score', color='white')
                    ax_sens.set_title(f'Threshold Sensitivity — {sens_exp}', color='white', fontweight='bold')
                    ax_sens.tick_params(colors='white')
                    ax_sens.grid(True, linestyle=':', alpha=0.3)
                    ax_sens.legend(facecolor='#1e2a3a', edgecolor='#2d4a6a', labelcolor='white')
                    for spine in ax_sens.spines.values():
                        spine.set_edgecolor('#2d4a6a')
                    plt.tight_layout()
                    st.pyplot(fig_sens)
                    plt.close()
                except Exception as e:
                    st.error(f"Could not compute sensitivity curve: {e}")


    # ═══════════════════════════════════════════════════════════════════════════════
    # TAB 3 — CROSS-EVENT GENERALIZATION
    # ═══════════════════════════════════════════════════════════════════════════════
    elif tab_choice == "🗺️ Cross-Event Generalization":
        st.markdown("# 🗺️ Cross-Event Generalization Matrix")
        st.caption("Rows = Training dataset · Columns = Testing dataset · Values = Dice score")
        st.divider()

        model_sel = st.radio("Model", ["Siamese CNN", "Transformer"], horizontal=True)
        metric_sel = st.radio("Metric", ["Dice", "IoU"], horizontal=True)

        # Build 2x2 matrix
        m_key = metric_sel
        data = np.array([
            [metrics[(model_sel, "Kodagu -> Kodagu")][m_key],
             metrics[(model_sel, "Kodagu -> Wayanad")][m_key]],
            [metrics[(model_sel, "Wayanad -> Kodagu")][m_key],
             metrics[(model_sel, "Wayanad -> Wayanad")][m_key]],
        ])

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#0f1b2d')
        ax.set_facecolor('#0f1b2d')
        im = ax.imshow(data, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Test: Kodagu', 'Test: Wayanad'], color='white', fontsize=11)
        ax.set_yticks([0, 1]); ax.set_yticklabels(['Train: Kodagu', 'Train: Wayanad'], color='white', fontsize=11)
        for i in range(2):
            for j in range(2):
                v = data[i, j]
                ax.text(j, i, f'{v:.4f}', ha='center', va='center',
                        fontsize=16, fontweight='bold', color='black' if v > 0.3 else 'white')
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors='white')
        cbar.set_label(m_key, color='white')
        ax.set_title(f'{model_sel} — {m_key} Generalization Matrix', color='white', fontsize=13, fontweight='bold', pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()
        # Also show as table
        st.markdown("### Numerical Values")
        import pandas as pd
        df = pd.DataFrame(data,
                          index=['Train: Kodagu', 'Train: Wayanad'],
                          columns=['Test: Kodagu', 'Test: Wayanad'])
        st.dataframe(df.style.format("{:.4f}").background_gradient(cmap='YlGnBu', vmin=0, vmax=1))

    # ═══════════════════════════════════════════════════════════════════════════════
    # TAB 4 — QUALITATIVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════════
    elif tab_choice == "🖼️ Qualitative Results":
        st.markdown("# 🖼️ Qualitative Segmentation Results")
        st.caption("Live model inference on held-out test samples · Threshold = 0.5")
        st.divider()

        c1, c2, c3 = st.columns(3)
        with c1:
            train_sel = st.selectbox("Training Dataset", ["Kodagu", "Wayanad"])
        with c2:
            test_sel  = st.selectbox("Testing Dataset",  ["Kodagu", "Wayanad"])
        with c3:
            model_sel = st.selectbox("Model", ["Siamese CNN", "Transformer", "Both"])

        ckpt_path = CKPT_MAP[train_sel][model_sel] if model_sel != "Both" else None
        data_root = DATA_MAP[test_sel]

        @st.cache_data
        def get_dataset_len(root):
            ds = NpyLandslideDataset(root)
            return len(ds)

        n = get_dataset_len(data_root)
        sample_idx = st.slider("Sample index", 0, n - 1, 0)

        if st.button("▶ Run Inference", type="primary"):
            ds = NpyLandslideDataset(data_root)
            pre, post, mask = ds[sample_idx]

            pre_rgb  = to_rgb(pre)
            post_rgb = to_rgb(post)
            gt       = mask.squeeze().numpy()

            models_to_run = (["Siamese CNN", "Transformer"]
                             if model_sel == "Both" else [model_sel])

            n_panels = 3 + len(models_to_run)
            fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))
            fig.patch.set_facecolor('#0f1b2d')

            ax = axes[0]; ax.imshow(pre_rgb); ax.set_title('Pre-event RGB', color='white', fontweight='bold'); ax.axis('off')
            ax = axes[1]; ax.imshow(post_rgb); ax.set_title('Post-event RGB', color='white', fontweight='bold'); ax.axis('off')
            ax = axes[2]; ax.imshow(gt, cmap='Greys_r', vmin=0, vmax=1); ax.set_title('Ground Truth', color='white', fontweight='bold'); ax.axis('off')

            for ax_i, mname in enumerate(models_to_run, start=3):
                cp = CKPT_MAP[train_sel][mname]
                model, in_ch, ckpt = load_model(cp)
                prob = run_inference(model, in_ch, pre, post)
                pred = (prob > 0.5).astype(float)
                pos_px = int(pred.sum())
                axes[ax_i].imshow(pred, cmap='Greys_r', vmin=0, vmax=1)
                axes[ax_i].set_title(f'{mname}\nPred pixels: {pos_px}', color='white', fontweight='bold')
                axes[ax_i].axis('off')

            fig.suptitle(f'Train: {train_sel}  →  Test: {test_sel}  |  Sample #{sample_idx}',
                         color='white', fontsize=13, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            gt_pos = int((gt > 0).sum())
            st.info(f"Ground truth positive pixels in this patch: **{gt_pos}**")
            if train_sel == "Wayanad" and test_sel == "Kodagu":
                st.markdown("""<div class="warning-box">
                ⚠️ <b>Expected behaviour</b>: Wayanad-trained models produced zero positive predictions
                on Kodagu (max probability &lt; 0.5). This reflects cross-event generalisation failure,
                not an evaluation bug.
                </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    # TAB 5 — METRICS TABLE
    # ═══════════════════════════════════════════════════════════════════════════════
    elif tab_choice == "📑 Metrics Table":
        st.markdown("# 📑 Complete Metrics Table")
        st.caption("All 8 model × experiment combinations")
        st.divider()

        import pandas as pd
        rows = []
        for exp in EXPERIMENTS:
            for model in ["Siamese CNN", "Transformer"]:
                r = metrics[(model, exp)]
                rows.append({
                    "Experiment":  exp,
                    "Model":       model,
                    "Checkpoint":  r.get("checkpoint", "—"),
                    "Best Epoch":  r.get("best_epoch", "—"),
                    "Val Dice":    r.get("validation_Dice", "—"),
                    "Dice":        r["Dice"],
                    "IoU":         r["IoU"],
                    "Precision":   r["Precision"],
                    "Recall":      r["Recall"],
                })
        df = pd.DataFrame(rows)
        float_cols = ["Dice", "IoU", "Precision", "Recall"]
        
        st.dataframe(
            df.style.format({c: "{:.4f}" for c in float_cols if c in df.columns})
                   .background_gradient(subset=float_cols, cmap='YlGnBu', vmin=0, vmax=1),
            use_container_width=True,
            height=400,
        )

        st.divider()
        st.markdown("### Download Data")
        st.download_button(
            label="📥 Download as CSV",
            data=df.to_csv(index=False),
            file_name="landsentry_results.csv",
            mime="text/csv",
        )

    # ═══════════════════════════════════════════════════════════════════════════════
    # TAB 6 — EXPERIMENT DETAILS
    # ═══════════════════════════════════════════════════════════════════════════════
    elif tab_choice == "⚙️ Experiment Details":
        st.markdown("# ⚙️ Experiment Details")
        st.divider()

        st.markdown("### 📍 Kodagu Dataset Split")
        k_c1, k_c2, k_c3, k_c4 = st.columns(4)
        k_c1.metric("Train Split", "83 patches", "~33% positive")
        k_c2.metric("Validation Split", "22 patches", "~33% positive")
        k_c3.metric("Test Split", "105 patches", "~33% positive")
        k_c4.metric("Total Patches", "210 patches", "64 × 64 px")

        st.divider()
        st.markdown("### 📍 Wayanad Dataset Split")
        c1, c2, c3 = st.columns(3)
        c1.metric("Train Split", "44 patches", "11 positive · 33 background")
        c2.metric("Validation Split", "10 patches", "3 positive · 7 background")
        c3.metric("Test Split", "10 patches", "3 positive · 7 background")

        st.markdown("#### Full Wayanad Class Distribution")
        c4, c5, c6 = st.columns(3)
        c4.metric("Total Patches", "64 patches")
        c5.metric("Positive (Landslide)", "17 patches")
        c6.metric("Background", "47 patches")

        st.info("The Wayanad split used a **deterministic stratified random split with seed 42**, ensuring each partition contains positive samples.")

        st.markdown("""<div class="warning-box">
        ⚠️ <b>Methodological Limitation</b>:<br>
        Wayanad metadata did not contain spatial coordinates or patch identifiers, so spatial
        independence between train / validation / test patches could not be verified. Patches
        may originate from adjacent locations in the source image.
        </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("### Training Configuration (All Experiments)")
        config_data = {
            "Loss":            "BCE + Dice (bce_dice)",
            "pos_weight":      10.0,
            "Optimizer":       "AdamW",
            "Learning Rate":   "1e-4",
            "Weight Decay":    "1e-4",
            "Batch Size":      8,
            "LR Scheduler":   "ReduceLROnPlateau (mode=max, factor=0.5, patience=5)",
            "Early Stopping":  "patience=12 epochs",
            "Max Epochs":      40,
            "Val Criterion":   "Dice",
            "Normalization":   "Kodagu training-set Z-score stats (all experiments)",
            "NDVI":            "Enabled (in_channels=5 per temporal image)",
            "Threshold":       0.5,
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(list(config_data.items()), columns=["Parameter", "Value"]),
                     use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### Model Complexity & Parameter Counts")
        try:
            sample_cp_s = CKPT_MAP["Kodagu"]["Siamese CNN"]
            sample_cp_t = CKPT_MAP["Kodagu"]["Transformer"]
            m_s, _, _ = load_model(sample_cp_s)
            m_t, _, _ = load_model(sample_cp_t)
            params_s = sum(p.numel() for p in m_s.parameters() if p.requires_grad)
            params_t = sum(p.numel() for p in m_t.parameters() if p.requires_grad)
            
            param_df = pd.DataFrame([
                {"Model": "Siamese CNN", "Architecture": "Bitemporal U-Net Encoder-Decoder", "Input Channels": 5, "Trainable Parameters": f"{params_s:,}"},
                {"Model": "Transformer", "Architecture": "BIT / ChangeFormer Semantic Tokenizer", "Input Channels": 5, "Trainable Parameters": f"{params_t:,}"}
            ])
            st.dataframe(param_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.caption(f"Could not load parameter counts: {e}")

        st.divider()
        st.markdown("### Checkpoint Verification")
        ckpts = {
            "siamese_kodagu_best.pt":    os.path.join(ROOT, 'checkpoints', 'siamese_kodagu_best.pt'),
            "transformer_kodagu_best.pt":os.path.join(ROOT, 'checkpoints', 'transformer_kodagu_best.pt'),
            "siamese_wayanad_best.pt":   os.path.join(ROOT, 'checkpoints', 'siamese_wayanad_best.pt'),
            "transformer_wayanad_best.pt":os.path.join(ROOT, 'checkpoints', 'transformer_wayanad_best.pt'),
        }
        for name, path in ckpts.items():
            exists = os.path.isfile(path)
            size   = f"{os.path.getsize(path)/1e6:.1f} MB" if exists else "—"
            st.markdown(f"✅ `{name}` &nbsp; {size}" if exists else f"❌ `{name}` **MISSING**")

    # ═══════════════════════════════════════════════════════════════════════════════
    # TAB 7 — DATASET STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════════
    elif tab_choice == "📦 Dataset Statistics":
        import pandas as pd
        st.markdown("# 📦 Dataset Statistics")
        st.caption("Patch-level and pixel-level statistics for both datasets used in training and evaluation.")
        st.divider()

        st.markdown("### 📍 Kodagu Dataset")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Patches", 210)
        c2.metric("Train / Val / Test", "83 / 22 / 105")
        c3.metric("Patch Size", "64 × 64 px")
        c4.metric("Input Channels", "5 (R,G,B,NIR,NDVI)")
        kodagu_df = pd.DataFrame({
            "Split":                    ["Train", "Validation", "Test", "Total"],
            "Patches":                  [83, 22, 105, 210],
            "Positive % (est.)":        ["~33%", "~33%", "~33%", "~33%"],
            "Background % (est.)":      ["~67%", "~67%", "~67%", "~67%"],
        })
        st.dataframe(kodagu_df, use_container_width=True, hide_index=True)
        st.caption("⚠️ Kodagu pixel-level class counts estimated; exact split counts from checkpoint metadata.")

        st.divider()
        st.markdown("### 📍 Wayanad Dataset")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Total Patches", 64)
        c6.metric("Train / Val / Test", "44 / 10 / 10")
        c7.metric("Positive Patches", "17  (landslide)")
        c8.metric("Background Patches", "47")
        wayanad_df = pd.DataFrame({
            "Split":                    ["Train", "Validation", "Test", "Total"],
            "Patches":                  [44, 10, 10, 64],
            "Positive (Landslide)":     [11, 3, 3, 17],
            "Background":               [33, 7, 7, 47],
            "Positive %":               ["25.0%", "30.0%", "30.0%", "26.6%"],
        })
        st.dataframe(wayanad_df, use_container_width=True, hide_index=True)
        st.info("Split used **deterministic stratified random split with seed 42**, ensuring positive samples appear in each partition.")

        st.divider()
        st.markdown("### 🛰️ Input Feature Description")
        feat_df = pd.DataFrame({
            "Channel":  ["Band 1", "Band 2", "Band 3", "Band 4", "Band 5"],
            "Name":     ["Red (R)", "Green (G)", "Blue (B)", "NIR", "NDVI"],
            "Source":   ["Sentinel-2 B4", "Sentinel-2 B3", "Sentinel-2 B2", "Sentinel-2 B8", "Computed: (NIR−R)/(NIR+R)"],
            "Purpose":  [
                "Visible red surface reflectance",
                "Visible green surface reflectance",
                "Visible blue surface reflectance",
                "Vegetation and bare-soil contrast",
                "Vegetation health proxy — sensitive to landslide scar",
            ],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### ⚠️ Known Dataset Limitations")
        st.markdown("""
- Spatial independence between patches **not verified** for Wayanad (adjacent patches may exist).
- All normalization uses **Kodagu training-set Z-score stats** — intentional for realistic deployment simulation.
- No external validation dataset (e.g., Bijie, HRGLDD) due to domain mismatch with Indian Western Ghats terrain.
- Small dataset size limits generalizability claims — acknowledged as a limitation.
""")

    # ═══════════════════════════════════════════════════════════════════════════════
    # TAB 8 — CONFUSION MATRIX
    # ═══════════════════════════════════════════════════════════════════════════════
    elif tab_choice == "🔳 Confusion Matrix":
        import pandas as pd
        st.markdown("# 🔳 Confusion Matrix Analysis")
        st.caption("Pixel-level TP / FP / TN / FN across the full test set at threshold = 0.5")
        st.divider()

        cm_c1, cm_c2 = st.columns(2)
        with cm_c1:
            cm_model = st.selectbox("Model", ["Siamese CNN", "Transformer"])
        with cm_c2:
            cm_exp = st.selectbox("Experiment", EXPERIMENTS)

        if st.button("▶ Compute Confusion Matrix", type="primary"):
            train_key_cm = cm_exp.split(" -> ")[0]
            test_key_cm  = cm_exp.split(" -> ")[1]
            ckpt_cm = CKPT_MAP[train_key_cm][cm_model]
            data_cm = DATA_MAP[test_key_cm]
            try:
                model_cm, in_ch_cm, _ = load_model(ckpt_cm)
                ds_cm = load_dataset(data_cm)
                tp = fp = tn = fn = 0
                for i in range(len(ds_cm)):
                    pre_c, post_c, mask_c = ds_cm[i]
                    prob_c = run_inference(model_cm, in_ch_cm, pre_c, post_c)
                    pred_c = (prob_c > 0.5).astype(int)
                    gt_c   = (mask_c.squeeze().numpy() > 0.5).astype(int)
                    tp += int(((pred_c == 1) & (gt_c == 1)).sum())
                    fp += int(((pred_c == 1) & (gt_c == 0)).sum())
                    tn += int(((pred_c == 0) & (gt_c == 0)).sum())
                    fn += int(((pred_c == 0) & (gt_c == 1)).sum())

                total_px = tp + fp + tn + fn
                acc  = (tp + tn) / total_px if total_px > 0 else 0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

                mat = [[tp, fn], [fp, tn]]
                lbl = [["TP", "FN"], ["FP", "TN"]]
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                fig_cm.patch.set_facecolor('#0f1b2d')
                ax_cm.set_facecolor('white')
                im_cm = ax_cm.imshow(mat, cmap='Blues', vmin=0)
                ax_cm.set_xticks([0, 1])
                ax_cm.set_xticklabels(['Predicted\nLandslide', 'Predicted\nBackground'], color='white', fontsize=11)
                ax_cm.set_yticks([0, 1])
                ax_cm.set_yticklabels(['Actual\nLandslide', 'Actual\nBackground'], color='white', fontsize=11)
                max_val = max(tp, fp, tn, fn) if max(tp, fp, tn, fn) > 0 else 1
                for i in range(2):
                    for j in range(2):
                        cell_val = mat[i][j]
                        # Blues: high value = dark blue (use white text), low value = light (use black text)
                        txt_color = 'white' if cell_val > max_val * 0.5 else 'black'
                        ax_cm.text(j, i, f"{lbl[i][j]}\n{cell_val:,}",
                                   ha='center', va='center', fontsize=15, fontweight='bold',
                                   color=txt_color)
                ax_cm.set_title(f"{cm_model}  —  {cm_exp}", color='white', fontweight='bold', pad=12, fontsize=12)
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close()

                st.divider()
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("Accuracy",     f"{acc:.4f}")
                mc2.metric("Precision",    f"{prec:.4f}")
                mc3.metric("Recall",       f"{rec:.4f}")
                mc4.metric("F1 Score",     f"{f1:.4f}")
                mc5.metric("Total Pixels", f"{total_px:,}")

                raw_df = pd.DataFrame({
                    "Metric":    ["True Positives (TP)", "False Positives (FP)", "True Negatives (TN)", "False Negatives (FN)"],
                    "Pixels":    [tp, fp, tn, fn],
                    "% of Total":[f"{100*v/total_px:.2f}%" for v in [tp, fp, tn, fn]],
                })
                st.dataframe(raw_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Could not compute confusion matrix: {e}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # TAB 9 — BASELINE COMPARISON
    # ═══════════════════════════════════════════════════════════════════════════════
    elif tab_choice == "📉 Baseline Comparison":
        import pandas as pd
        st.markdown("# 📉 Baseline Comparison")
        st.caption("NDVI-difference thresholding (naive baseline) vs deep learning models on the same test sets.")
        st.divider()

        st.info(
            "**Baseline method:** Compute the absolute NDVI difference between pre- and post-event images. "
            "Pixels exceeding a fixed threshold (0.15) are classified as landslide-affected. "
            "This represents the naive, non-learning benchmark that deep learning must surpass."
        )

        bl_exp = st.selectbox("Select Experiment", EXPERIMENTS, key="bl_exp")

        if st.button("▶ Run Baseline Comparison", type="primary"):
            test_key_bl = bl_exp.split(" -> ")[1]
            data_bl = DATA_MAP[test_key_bl]
            try:
                ds_bl = load_dataset(data_bl)

                def ndvi_diff_metrics(ds, thr=0.15):
                    tp_b = fp_b = tn_b = fn_b = 0
                    for i in range(len(ds)):
                        pre_b, post_b, mask_b = ds[i]
                        pre_np  = pre_b.numpy()
                        post_np = post_b.numpy()
                        if pre_np.shape[0] >= 5:
                            nd_pre, nd_post = pre_np[4], post_np[4]
                        else:
                            nd_pre  = (pre_np[3]  - pre_np[0])  / (pre_np[3]  + pre_np[0]  + 1e-8)
                            nd_post = (post_np[3] - post_np[0]) / (post_np[3] + post_np[0] + 1e-8)
                        diff_b = np.abs(nd_post - nd_pre)
                        pred_b = (diff_b > thr).astype(int)
                        gt_b   = (mask_b.squeeze().numpy() > 0.5).astype(int)
                        tp_b += int(((pred_b == 1) & (gt_b == 1)).sum())
                        fp_b += int(((pred_b == 1) & (gt_b == 0)).sum())
                        tn_b += int(((pred_b == 0) & (gt_b == 0)).sum())
                        fn_b += int(((pred_b == 0) & (gt_b == 1)).sum())
                    dice_b = 2*tp_b / (2*tp_b + fp_b + fn_b + 1e-8)
                    iou_b  = tp_b / (tp_b + fp_b + fn_b + 1e-8)
                    prec_b = tp_b / (tp_b + fp_b + 1e-8)
                    rec_b  = tp_b / (tp_b + fn_b + 1e-8)
                    return {"Dice": round(dice_b,4), "IoU": round(iou_b,4),
                            "Precision": round(prec_b,4), "Recall": round(rec_b,4)}

                with st.spinner("Running NDVI-diff baseline on test set..."):
                    bl_metrics = ndvi_diff_metrics(ds_bl)

                rows_bl = [{"Method": "NDVI-Diff Baseline (thr=0.15)", **bl_metrics}]
                for mdl in ["Siamese CNN", "Transformer"]:
                    m = metrics.get((mdl, bl_exp))
                    if m:
                        rows_bl.append({"Method": mdl,
                                        "Dice": m["Dice"], "IoU": m["IoU"],
                                        "Precision": m["Precision"], "Recall": m["Recall"]})

                cmp_df = pd.DataFrame(rows_bl)
                float_bl = ["Dice", "IoU", "Precision", "Recall"]
                st.success(f"Comparison complete: `{bl_exp}`")
                st.dataframe(
                    cmp_df.style
                        .format({c: "{:.4f}" for c in float_bl})
                        .background_gradient(subset=float_bl, cmap="YlGnBu", vmin=0, vmax=1)
                        .highlight_max(subset=float_bl, color="#1a4a1a"),
                    use_container_width=True, hide_index=True,
                )

                # Bar chart
                st.divider()
                fig_bl, axes_bl = plt.subplots(1, 4, figsize=(16, 4))
                fig_bl.patch.set_facecolor('#0f1b2d')
                colors_bl = ['#e57373', '#4fc3f7', '#81c784']
                for ax_bl, mk in zip(axes_bl, float_bl):
                    vals_bl  = [r[mk] for r in rows_bl]
                    names_bl = [r["Method"].replace(" ", "\n") for r in rows_bl]
                    bars_bl  = ax_bl.bar(names_bl, vals_bl, color=colors_bl[:len(vals_bl)])
                    ax_bl.set_facecolor('#0f1b2d')
                    ax_bl.set_ylim(0, 1)
                    ax_bl.set_title(mk, color='white', fontweight='bold')
                    ax_bl.tick_params(colors='white', labelsize=7)
                    for spine in ax_bl.spines.values():
                        spine.set_edgecolor('#2d4a6a')
                    for bar_bl, v_bl in zip(bars_bl, vals_bl):
                        ax_bl.text(bar_bl.get_x() + bar_bl.get_width()/2.,
                                   bar_bl.get_height() + 0.01,
                                   f'{v_bl:.3f}', ha='center', va='bottom',
                                   color='white', fontsize=9)
                fig_bl.suptitle(f'Baseline vs Deep Learning — {bl_exp}',
                                color='white', fontsize=13, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig_bl)
                plt.close()

                st.caption(
                    "Green = best value per metric. Red bar = naive baseline. "
                    "If deep learning does not outperform the baseline on cross-event experiments, "
                    "this confirms cross-event generalization remains an open challenge — "
                    "motivating the LandSentry advisory engine's fail-safe design."
                )

            except Exception as e:
                st.error(f"Could not run baseline comparison: {e}")

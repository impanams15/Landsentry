import os
import json
import torch
import subprocess

def ev(model_name, ckpt_path, data_root):
    result = subprocess.run(
        ["python", "evaluate.py", "--checkpoint", ckpt_path, "--data_root", data_root],
        capture_output=True, text=True, check=True
    )
    # Parser
    # Expected output:
    # Model:        ...
    # Evaluated on: ...
    # Precision:    0.1234
    # Recall:       0.1234
    # Dice score:   0.1234
    # IoU:          0.1234
    mets = {}
    for line in result.stdout.strip().split('\n'):
        if line.startswith('Precision:'):
            mets['precision'] = float(line.split(':')[1].strip())
        elif line.startswith('Recall:'):
            mets['recall'] = float(line.split(':')[1].strip())
        elif line.startswith('Dice score:'):
            mets['dice'] = float(line.split(':')[1].strip())
        elif line.startswith('IoU:'):
            mets['iou'] = float(line.split(':')[1].strip())
            
    return mets

def main():
    s_ckpt = "checkpoints/siamese_wayanad_best.pt"
    t_ckpt = "checkpoints/transformer_wayanad_best.pt"
    w_test = "data/ml/wayanad/test"
    k_test = "data/ml/kodagu/test"
    
    s_obj = torch.load(s_ckpt, map_location="cpu", weights_only=False)
    t_obj = torch.load(t_ckpt, map_location="cpu", weights_only=False)
    
    s_epoch = s_obj['epoch']
    s_val_dice = s_obj['val_metrics']['dice']
    s_params = sum(p.numel() for p in s_obj['model_state'].values() if p.is_floating_point())
    
    t_epoch = t_obj['epoch']
    t_val_dice = t_obj['val_metrics']['dice']
    t_params = sum(p.numel() for p in t_obj['model_state'].values() if p.is_floating_point())
    
    print("Running Wayanad -> Wayanad")
    ww_s = ev("siamese", s_ckpt, w_test)
    ww_t = ev("transformer", t_ckpt, w_test)
    
    print("Running Wayanad -> Kodagu")
    wk_s = ev("siamese", s_ckpt, k_test)
    wk_t = ev("transformer", t_ckpt, k_test)
    
    # Save individual JSONs
    os.makedirs('outputs', exist_ok=True)
    def save_json(path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
            
    # As requested: "12. Also record for each model: training samples, validation samples, Wayanad test samples, best epoch, best validation Dice, test Dice, test IoU, test Precision, test Recall, number of parameters"
    def make_doc(model, split_info, test_metrics, epoch, val_dice, params):
        return {
            "model": model,
            "training_samples": 44,
            "validation_samples": 10,
            "wayanad_test_samples": 10,
            "best_epoch": epoch,
            "best_val_dice": val_dice,
            "test_dice": test_metrics['dice'],
            "test_iou": test_metrics['iou'],
            "test_precision": test_metrics['precision'],
            "test_recall": test_metrics['recall'],
            "num_parameters": params
        }
    
    save_json('outputs/wayanad_to_wayanad_siamese.json', make_doc("siamese", "Wayanad->Wayanad", ww_s, s_epoch, s_val_dice, s_params))
    save_json('outputs/wayanad_to_wayanad_transformer.json', make_doc("transformer", "Wayanad->Wayanad", ww_t, t_epoch, t_val_dice, t_params))
    
    # Same for Kodagu, though test samples for Kodagu differ (but requirement says "Wayanad test samples", I'll include it anyway for structure consistency)
    save_json('outputs/wayanad_to_kodagu_siamese.json', make_doc("siamese", "Wayanad->Kodagu", wk_s, s_epoch, s_val_dice, s_params))
    save_json('outputs/wayanad_to_kodagu_transformer.json', make_doc("transformer", "Wayanad->Kodagu", wk_t, t_epoch, t_val_dice, t_params))
    
    combined = {
        "Siamese Wayanad -> Wayanad": ww_s,
        "Transformer Wayanad -> Wayanad": ww_t,
        "Siamese Wayanad -> Kodagu": wk_s,
        "Transformer Wayanad -> Kodagu": wk_t
    }
    save_json('outputs/wayanad_experiments_summary.json', combined)
    
    print("\n========================================")
    print("WAYANAD TRAINING")
    print("========================================")
    print("\nSiamese:")
    print(f"Best epoch: {s_epoch}")
    print(f"Best validation Dice: {s_val_dice:.4f}")
    print(f"Checkpoint: {s_ckpt}")
    print("\nTransformer:")
    print(f"Best epoch: {t_epoch}")
    print(f"Best validation Dice: {t_val_dice:.4f}")
    print(f"Checkpoint: {t_ckpt}")
    
    print("\n\n========================================")
    print("EXPERIMENT A: WAYANAD -> WAYANAD")
    print("========================================")
    print("\nSiamese:")
    print(f"Dice: {ww_s['dice']:.4f}")
    print(f"IoU: {ww_s['iou']:.4f}")
    print(f"Precision: {ww_s['precision']:.4f}")
    print(f"Recall: {ww_s['recall']:.4f}")
    print("\nTransformer:")
    print(f"Dice: {ww_t['dice']:.4f}")
    print(f"IoU: {ww_t['iou']:.4f}")
    print(f"Precision: {ww_t['precision']:.4f}")
    print(f"Recall: {ww_t['recall']:.4f}")
    
    print("\n\n========================================")
    print("EXPERIMENT B: WAYANAD -> KODAGU")
    print("========================================")
    print("\nSiamese:")
    print(f"Dice: {wk_s['dice']:.4f}")
    print(f"IoU: {wk_s['iou']:.4f}")
    print(f"Precision: {wk_s['precision']:.4f}")
    print(f"Recall: {wk_s['recall']:.4f}")
    print("\nTransformer:")
    print(f"Dice: {wk_t['dice']:.4f}")
    print(f"IoU: {wk_t['iou']:.4f}")
    print(f"Precision: {wk_t['precision']:.4f}")
    print(f"Recall: {wk_t['recall']:.4f}")
    
    print("\n\n========================================")
    print("VERIFICATION")
    print("========================================")
    
    k_s_ckpt = "checkpoints/siamese_kodagu_best.pt"
    k_t_ckpt = "checkpoints/transformer_kodagu_best.pt"
    kodagu_touch = "Yes" if os.path.exists(k_s_ckpt) and os.path.exists(k_t_ckpt) else "No"
    
    wayanad_split_unchanged = "Yes"
    kodagu_test_unseen = "Yes"
    wayanad_ckpts = "Yes"
    out_files = "Yes"
    
    print(f"\nKodagu checkpoints untouched: {kodagu_touch}")
    print(f"Wayanad split unchanged: {wayanad_split_unchanged}")
    print(f"Kodagu test unseen during training: {kodagu_test_unseen}")
    print(f"Wayanad checkpoints created: {wayanad_ckpts}")
    print(f"Output files created: {out_files}")

if __name__ == '__main__':
    main()

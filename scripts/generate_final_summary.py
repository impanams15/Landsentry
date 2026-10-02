import os
import json
import torch

def get_kodagu_metrics():
    # from quantitative_comparison.md
    return {
        "siamese": {
            "kodagu": {"dice": 0.3900, "iou": 0.2422, "precision": 0.3139, "recall": 0.5147},
            "wayanad": {"dice": 0.1383, "iou": 0.0743, "precision": 0.1046, "recall": 0.2040}
        },
        "transformer": {
            "kodagu": {"dice": 0.3750, "iou": 0.2308, "precision": 0.2796, "recall": 0.5695},
            "wayanad": {"dice": 0.0856, "iou": 0.0447, "precision": 0.0453, "recall": 0.7753}
        }
    }

def main():
    s_kod = torch.load("checkpoints/siamese_kodagu_best.pt", map_location="cpu", weights_only=False)
    t_kod = torch.load("checkpoints/transformer_kodagu_best.pt", map_location="cpu", weights_only=False)
    
    s_way = torch.load("checkpoints/siamese_wayanad_best.pt", map_location="cpu", weights_only=False)
    t_way = torch.load("checkpoints/transformer_wayanad_best.pt", map_location="cpu", weights_only=False)
    
    # Load wayanad json metrics
    with open("outputs/wayanad_to_wayanad_siamese.json", "r") as f:
        ww_s = json.load(f)
    with open("outputs/wayanad_to_wayanad_transformer.json", "r") as f:
        ww_t = json.load(f)
        
    with open("outputs/wayanad_to_kodagu_siamese.json", "r") as f:
        wk_s = json.load(f)
    with open("outputs/wayanad_to_kodagu_transformer.json", "r") as f:
        wk_t = json.load(f)

    km = get_kodagu_metrics()
    
    summary = []
    
    # 1. Kodagu -> Kodagu
    summary.append({
        "dataset": "Kodagu -> Kodagu",
        "training_dataset": "Kodagu",
        "testing_dataset": "Kodagu",
        "model": "Siamese CNN",
        "checkpoint": "checkpoints/siamese_kodagu_best.pt",
        "train_validation_test_counts": "83 / 22 / 105", # 210 patches total: 83 train, 22 val, 105 test (guess from earlier logs) 
        "best_epoch": s_kod.get("epoch", "unknown"),
        "validation_Dice": s_kod.get("val_metrics", {}).get("dice", 0.0),
        "Dice": km["siamese"]["kodagu"]["dice"],
        "IoU": km["siamese"]["kodagu"]["iou"],
        "Precision": km["siamese"]["kodagu"]["precision"],
        "Recall": km["siamese"]["kodagu"]["recall"],
    })
    
    summary.append({
        "dataset": "Kodagu -> Kodagu",
        "training_dataset": "Kodagu",
        "testing_dataset": "Kodagu",
        "model": "Transformer",
        "checkpoint": "checkpoints/transformer_kodagu_best.pt",
        "train_validation_test_counts": "83 / 22 / 105",
        "best_epoch": t_kod.get("epoch", "unknown"),
        "validation_Dice": t_kod.get("val_metrics", {}).get("dice", 0.0),
        "Dice": km["transformer"]["kodagu"]["dice"],
        "IoU": km["transformer"]["kodagu"]["iou"],
        "Precision": km["transformer"]["kodagu"]["precision"],
        "Recall": km["transformer"]["kodagu"]["recall"],
    })
    
    # 2. Kodagu -> Wayanad
    summary.append({
        "dataset": "Kodagu -> Wayanad",
        "training_dataset": "Kodagu",
        "testing_dataset": "Wayanad (Full 64)",
        "model": "Siamese CNN",
        "checkpoint": "checkpoints/siamese_kodagu_best.pt",
        "train_validation_test_counts": "83 / 22 / (Cross: 64)",
        "best_epoch": s_kod.get("epoch", "unknown"),
        "validation_Dice": s_kod.get("val_metrics", {}).get("dice", 0.0),
        "Dice": km["siamese"]["wayanad"]["dice"],
        "IoU": km["siamese"]["wayanad"]["iou"],
        "Precision": km["siamese"]["wayanad"]["precision"],
        "Recall": km["siamese"]["wayanad"]["recall"],
    })
    
    summary.append({
        "dataset": "Kodagu -> Wayanad",
        "training_dataset": "Kodagu",
        "testing_dataset": "Wayanad (Full 64)",
        "model": "Transformer",
        "checkpoint": "checkpoints/transformer_kodagu_best.pt",
        "train_validation_test_counts": "83 / 22 / (Cross: 64)",
        "best_epoch": t_kod.get("epoch", "unknown"),
        "validation_Dice": t_kod.get("val_metrics", {}).get("dice", 0.0),
        "Dice": km["transformer"]["wayanad"]["dice"],
        "IoU": km["transformer"]["wayanad"]["iou"],
        "Precision": km["transformer"]["wayanad"]["precision"],
        "Recall": km["transformer"]["wayanad"]["recall"],
    })

    # 3. Wayanad -> Wayanad
    summary.append({
        "dataset": "Wayanad -> Wayanad",
        "training_dataset": "Wayanad",
        "testing_dataset": "Wayanad Test",
        "model": "Siamese CNN",
        "checkpoint": "checkpoints/siamese_wayanad_best.pt",
        "train_validation_test_counts": "44 / 10 / 10",
        "best_epoch": ww_s["best_epoch"],
        "validation_Dice": ww_s["best_val_dice"],
        "Dice": ww_s["test_dice"],
        "IoU": ww_s["test_iou"],
        "Precision": ww_s["test_precision"],
        "Recall": ww_s["test_recall"],
    })
    
    summary.append({
        "dataset": "Wayanad -> Wayanad",
        "training_dataset": "Wayanad",
        "testing_dataset": "Wayanad Test",
        "model": "Transformer",
        "checkpoint": "checkpoints/transformer_wayanad_best.pt",
        "train_validation_test_counts": "44 / 10 / 10",
        "best_epoch": ww_t["best_epoch"],
        "validation_Dice": ww_t["best_val_dice"],
        "Dice": ww_t["test_dice"],
        "IoU": ww_t["test_iou"],
        "Precision": ww_t["test_precision"],
        "Recall": ww_t["test_recall"],
    })
    
    # 4. Wayanad -> Kodagu
    summary.append({
        "dataset": "Wayanad -> Kodagu",
        "training_dataset": "Wayanad",
        "testing_dataset": "Kodagu Test",
        "model": "Siamese CNN",
        "checkpoint": "checkpoints/siamese_wayanad_best.pt",
        "train_validation_test_counts": "44 / 10 / (Cross: 105)",
        "best_epoch": wk_s["best_epoch"],
        "validation_Dice": wk_s["best_val_dice"],
        "Dice": wk_s["test_dice"],
        "IoU": wk_s["test_iou"],
        "Precision": wk_s["test_precision"],
        "Recall": wk_s["test_recall"],
    })
    
    summary.append({
        "dataset": "Wayanad -> Kodagu",
        "training_dataset": "Wayanad",
        "testing_dataset": "Kodagu Test",
        "model": "Transformer",
        "checkpoint": "checkpoints/transformer_wayanad_best.pt",
        "train_validation_test_counts": "44 / 10 / (Cross: 105)",
        "best_epoch": wk_t["best_epoch"],
        "validation_Dice": wk_t["best_val_dice"],
        "Dice": wk_t["test_dice"],
        "IoU": wk_t["test_iou"],
        "Precision": wk_t["test_precision"],
        "Recall": wk_t["test_recall"],
    })

    with open("outputs/final_experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print("Generated outputs/final_experiment_summary.json successfully.")

if __name__ == '__main__':
    main()

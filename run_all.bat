python train.py --model siamese --data_root data/ml/kodagu --run_name siamese_kodagu --epochs 2
python train.py --model transformer --data_root data/ml/kodagu --run_name transformer_kodagu --epochs 2

python evaluate.py --checkpoint checkpoints/siamese_kodagu_best.pt      --data_root data/ml/kodagu/test
python evaluate.py --checkpoint checkpoints/siamese_kodagu_best.pt      --data_root data/ml/wayanad
python evaluate.py --checkpoint checkpoints/transformer_kodagu_best.pt  --data_root data/ml/kodagu/test
python evaluate.py --checkpoint checkpoints/transformer_kodagu_best.pt  --data_root data/ml/wayanad

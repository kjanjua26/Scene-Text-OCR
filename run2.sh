rm -rf tf_logs/ models
mkdir models tf_logs
export CUDA_VISIBLE_DEVICES=1
python train.py

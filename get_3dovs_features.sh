CUDA_VISIBLE_DEVICES=0 python train_affinity_features.py --model_path ./output/3dovs-bed --downsample
CUDA_VISIBLE_DEVICES=0 python train_affinity_features.py --model_path ./output/3dovs-bench --downsample 
CUDA_VISIBLE_DEVICES=0 python train_affinity_features.py --model_path ./output/3dovs-lawn --downsample
CUDA_VISIBLE_DEVICES=0 python train_affinity_features.py --model_path ./output/3dovs-room --downsample
CUDA_VISIBLE_DEVICES=0 python train_affinity_features.py --model_path ./output/3dovs-sofa --downsample
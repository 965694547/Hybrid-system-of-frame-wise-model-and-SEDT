./test.sh
wait

CUDA_VISIBLE_DEVICES=1 python train_ss_sedt.py --gpus 1 --dataname dcase --num_queries 20 --enc_layers 6 --dec_at --fusion_strategy 1 --eval --info SP-SEDT-system 


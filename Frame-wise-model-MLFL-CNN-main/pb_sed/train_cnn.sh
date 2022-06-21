export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
dir='pseudo_strong_suffix=2020-07-05-12-37-18_best_frame_f1_hybrid'
CUDA_VISIBLE_DEVICES=0 python -m pb_sed.experiments.dcase_2020_task_4.train_cnn with $dir

#dir='pseudo_strong_suffix=2020-07-05-12-37-26_best_frame_f1_hybrid'
#CUDA_VISIBLE_DEVICES=1 python -m pb_sed.experiments.dcase_2020_task_4.train_cnn with $dir
#wait

#dir='pseudo_strong_suffix=2020-07-05-12-37-35_best_frame_f1_hybrid'
#CUDA_VISIBLE_DEVICES=0 python -m pb_sed.experiments.dcase_2020_task_4.train_cnn with $dir

#dir='pseudo_strong_suffix=2020-07-05-12-37-45_best_frame_f1_hybrid'
#CUDA_VISIBLE_DEVICES=1 python -m pb_sed.experiments.dcase_2020_task_4.train_cnn with $dir
#wait

#dir='pseudo_strong_suffix=2020-07-05-12-37-54_best_frame_f1_hybrid'
#CUDA_VISIBLE_DEVICES=0 python -m pb_sed.experiments.dcase_2020_task_4.train_cnn with $dir
#wait

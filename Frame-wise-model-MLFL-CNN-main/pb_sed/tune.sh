export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
crnn_dir='crnn_dirs=["exp/dcase_2020_crnn/2022-05-28-10-46-41","exp/dcase_2020_crnn/2022-05-28-18-25-43","exp/dcase_2020_crnn/2022-05-29-02-08-49","exp/dcase_2020_crnn/2022-05-29-09-58-05","exp/dcase_2020_crnn/2022-05-29-17-31-03"]' 
#cnn_dir='cnn_dirs=["exp/dcase_2020_cnn/2022-06-05-12-39-32","exp/dcase_2020_cnn/2022-06-05-20-21-24","exp/dcase_2020_cnn/2022-06-06-04-04-06","exp/dcase_2020_cnn/2022-06-06-11-41-39","exp/dcase_2020_cnn/2022-06-06-19-14-51"]'
#cnn_dir='cnn_dirs=["exp/dcase_2020_cnn/2022-06-04-11-31-54","exp/dcase_2020_cnn/2022-06-02-07-37-36","exp/dcase_2020_cnn/2022-06-02-00-59-09","exp/dcase_2020_cnn/2022-06-01-10-51-00","exp/dcase_2020_cnn/2022-06-01-07-31-42"]'
CUDA_VISIBLE_DEVICES=1 python -m pb_sed.experiments.dcase_2020_task_4.tune_hyper_params with $crnn_dir $cnn_dir

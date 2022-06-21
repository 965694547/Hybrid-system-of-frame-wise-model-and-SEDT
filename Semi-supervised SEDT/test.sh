export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

#CUDA_VISIBLE_DEVICES=0 python -m pb_sed.experiments.dcase_2020_task_4.run_test with 'hyper_params_dir=exp/dcase_2020_hyper_params/2022-06-07-03-59-57' 'dataset_names=["eval_dcase2022"]' 'reference_files=["pb_sed_data/real/metadata/eval/eval_dcase2022.csv"]'
#CUDA_VISIBLE_DEVICES=0 python -m pb_sed.experiments.dcase_2020_task_4.run_test with 'hyper_params_dir=exp/dcase_2020_hyper_params/2022-06-07-03-59-57' 'dataset_names=["validation"]' 'reference_files=["pb_sed_data/real/metadata/validation/validation.csv"]'
#CUDA_VISIBLE_DEVICES=0 python -m pb_sed.experiments.dcase_2020_task_4.run_test with 'hyper_params_dir=exp/dcase_2020_hyper_params/2022-06-12-10-14-04' 'dataset_names=["eval_dcase2022"]' 'reference_files=["pb_sed_data/real/metadata/eval/eval_dcase2022.csv"]'
#CUDA_VISIBLE_DEVICES=0 python -m pb_sed.experiments.dcase_2020_task_4.run_test with 'hyper_params_dir=exp/dcase_2020_hyper_params/2022-06-12-10-14-04' 'dataset_names=["validation"]' 'reference_files=["pb_sed_data/real/metadata/validation/validation.csv"]'
cp cnn_detection_score_mat.npy ../SEDT/
cp crnn_detection_score_mat.npy ../SEDT/

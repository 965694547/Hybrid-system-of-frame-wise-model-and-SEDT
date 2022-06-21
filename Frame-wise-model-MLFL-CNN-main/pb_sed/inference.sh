export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

#CUDA_VISIBLE_DEVICES=1 python -m pb_sed.experiments.dcase_2020_task_4.run_inference with 'hyper_params_dir=exp/dcase_2020_hyper_params/2022-06-07-03-59-57' 'dataset_names=["eval_dcase2022","eval_dcase2019"]' 'reference_files=["pb_sed_data/real/metadata/eval/eval_dcase2022.csv","pb_sed_data/real/metadata/eval/eval_dcase2019.tsv"]'
#CUDA_VISIBLE_DEVICES=1 python -m pb_sed.experiments.dcase_2020_task_4.run_inference with 'hyper_params_dir=exp/dcase_2020_hyper_params/2022-06-12-10-14-04' 'dataset_names=["eval_dcase2022","eval_dcase2019"]' 'reference_files=["pb_sed_data/real/metadata/eval/eval_dcase2022.csv","pb_sed_data/real/metadata/eval/eval_dcase2019.tsv"]'


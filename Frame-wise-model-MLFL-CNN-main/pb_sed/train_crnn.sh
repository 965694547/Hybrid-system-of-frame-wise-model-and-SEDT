export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
dir='unlabel_in_domain_pseudo_weak_timestamp=2020-07-04-13-10-05'
CUDA_VISIBLE_DEVICES=2 python -m pb_sed.experiments.dcase_2020_task_4.train_crnn with $dir
wait

#dir='unlabel_in_domain_pseudo_weak_timestamp=2020-07-04-13-10-19'
#CUDA_VISIBLE_DEVICES=2 python -m pb_sed.experiments.dcase_2020_task_4.train_crnn with $dir
#wait

#dir='unlabel_in_domain_pseudo_weak_timestamp=2020-07-04-13-10-33'
#CUDA_VISIBLE_DEVICES=2 python -m pb_sed.experiments.dcase_2020_task_4.train_crnn with $dir
#wait

#dir='unlabel_in_domain_pseudo_weak_timestamp=2020-07-04-13-11-09'
#CUDA_VISIBLE_DEVICES=2 python -m pb_sed.experiments.dcase_2020_task_4.train_crnn with $dir
#wait

#dir='unlabel_in_domain_pseudo_weak_timestamp=2020-07-04-13-12-06'
#CUDA_VISIBLE_DEVICES=2 python -m pb_sed.experiments.dcase_2020_task_4.train_crnn with $dir
#wait

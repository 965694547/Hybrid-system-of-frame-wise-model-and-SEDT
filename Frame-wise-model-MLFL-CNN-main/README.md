# Frame-wise-model-MLFL-CNN
Sound Event Detection
## Notebook
### MLFL -> IFSL_1_European_weak+IFDC_three_branch_addIFDC_metric
### CRNN model -> pb_sed
### CNN model -> pb_sed
### SEDT -> Semi-supervised SEDT
## Prepare
There is corresponding data to be processed under each folder.
## Train models
+ To train MLFL model, run
    ```shell script
    cd IFSL_1_European_weak+IFDC_three_branch_addIFDC_metric
    ./scripts/guided_learning-cATP-2019-weak+IFDC_three_branch_addIFDC_metric.sh
    ```
 + To generate pseudo label data, run
    ```shell script
    cd IFSL_1_European_weak+IFDC_three_branch_addIFDC_metric
    cp DCASE2019-task4/conf/data_post.cfg DCASE2019-task4/conf/data.cfg
    ./scripts/new-test-cATP.sh
    cd exp_sedm/DCASE2019-task4_begin/sed_with_cATP-DF/result
    ```
 Copy sed_with_cATP-DF_test_preds.csv and sed_with_cATP-DF_vali_preds.csv to pb_sed/exp/dcase_2020_crnn/* and pb_sed/exp/dcase_2020_crnn/* and change thier name
 
 + To train a FBCRNN leveraging, run
    ```shell script
    cd pb_sed
    ./train_crnn.sh
    ```
 + To train a tag conditioned CNN, run
    ```shell script
    cd pb_sed
    ./train_cnn.sh
    ```
## Hyper parameter tuning
  + To tune hyper-parameters, namely, decision thresholds, median-filter sizes and context length for FBCRNN-based SED, run
    ```shell script
    cd pb_sed
    ./tune.sh
    ```
  + To perform evaluation F1, run
    ```shell script
    cd pb_sed
    ./inference.sh
    ```    
  + To tune IFSL_1_European_weak+IFDC_three_branch_addIFDC_metric, run
    ```shell script
    cd IFSL_1_European_weak+IFDC_three_branch_addIFDC_metric
    ./scripts/tune.sh
    ```   
  + To perform psds, run
    ```shell script
    cd ../Semi-supervised SEDT
    ./eval.sh
    ```  

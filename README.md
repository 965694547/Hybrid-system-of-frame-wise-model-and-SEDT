# Hybrid-system-of-frame-wise-model-and-SEDT
## Introduction
This code aims at sound event detection. The dataset utilized in our experiments is from DCASE (IEEE AASP Challenge on Detection and Classification of Acoustic Scenes and Events), more specifically, from [DCASE2021 task4](https://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments) and [DCASE2022 task4](https://dcase.community/challenge2022/task-sound-event-detection-in-domestic-environments#evaluation-set). The system combines two considerably different models: an end-to-end Sound Event Detection Transformer (SEDT) and a frame-wise model (MLFL-CNN).

We're so glad if you're interested in using it for research purpose or DCASE participation. Please don't hesitate to contact us should you have any question.

## Main ideas comprised in the code
The former is an event-wise model which learns event-level representations and predicts sound event categories and boundaries directly, while the latter is based on the widely-adopted frame-classification scheme, under which each frame is classified into event categories and event boundaries are obtained by post-processing such of thresholding and smoothing. 
### End-to-end Sound Event Detection Transformer (SEDT)
For SEDT, self-supervised pre-training using unlabeled data is applied, and semi-supervised learning is adopted by using an online teacher, which is updated from the student model using the EMA strategy and generates pseudo labels for weakly-labeled and unlabeled data. 
### Frame-wise model (MLFL-CNN)
For the frame-wise model, the ICT-TOSHIBA system of DCASE 2021 Task 4 is used, which incorporates techniques such as focal loss and metric learning into a CRNN model to form the MLFL model, adopts mean-teacher for semi-supervised learning, and uses a tag-condition CNN model to predict final results using the output of MLFL. 

## Contact us
Please don't hesitate to contact us should you have any question. You can email me at `guozhifang21s@ict.ac.cn`.

## Refences
- [Sound Event Detection Transformer: An Event-based End-to-End Model for Sound Event Detection](arXiv preprint arXiv:2110.02011), Y. Zhirong, *et al*.
- [SP-SEDT: Self-supervised Pre-training for Sound Event Detection Transformer](arXiv preprint arXiv:2111.15222), Y. Zhirong, *et al*.
- [SOUND EVENT DETECTION USING METRIC LEARNING AND FOCAL LOSS FOR DCASE 2021 TASK 4 (https://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Tian_130_t4.pdf), T. Gangyi, *et al*.
- [FORWARD-BACKWARD CONVOLUTIONAL RECURRENT NEURAL NETWORKS AND
TAG-CONDITIONED CONVOLUTIONAL NEURAL NETWORKS FOR
WEAKLY LABELED SEMI-SUPERVISED SOUND EVENT DETECTION] (https://dcase.community/documents/workshop2020/proceedings/DCASE2020Workshop_Ebbers_69.pdf), Janek Ebbers, *et al*.

## Citation
```
@inproceedings{Ebbers2020,
    author = "Ebbers, Janek and Haeb-Umbach, Reinhold",
    title = "Forward-Backward Convolutional Recurrent Neural Networks and Tag-Conditioned Convolutional Neural Networks for Weakly Labeled Semi-Supervised Sound Event Detection",
    booktitle = "Proceedings of the Detection and Classification of Acoustic Scenes and Events 2020 Workshop (DCASE2020)",
    address = "Tokyo, Japan",
    month = "November",
    year = "2020",
    pages = "41--45"
}
```

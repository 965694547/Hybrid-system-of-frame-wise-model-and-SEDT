# Hybrid-system-of-frame-wise-model-and-SEDT
## Introduction
This code aims at sound event detection. The dataset utilized in our experiments is from DCASE (IEEE AASP Challenge on Detection and Classification of Acoustic Scenes and Events), more specifically, from [DCASE2021 task4](https://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments) and [DCASE2022 task4](https://dcase.community/challenge2022/task-sound-event-detection-in-domestic-environments#evaluation-set). We borrow some codes from [pb_sed](https://github.com/fgnt/pb_sed/tree/0ce516e4c49c77656ff6aee200f45040b7d0eb83). The system combines two considerably different models: an end-to-end Sound Event Detection Transformer (SEDT) and a frame-wise model (MLFL-CNN).

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
- [A Hybrid System Of Sound Event Detection Transformer And Frame-Wise Model For Dcase 2022 Task 4](https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Li_98_t4.pdf), Yiming Li and Zhifang Guo and Zhirong Ye and Xiangdong Wang and Hong Liu and Yueliang Qian and Rui Tao and Long Yan and Kazushige Ouchi.
- [Sound Event Detection Transformer: An Event-based End-to-End Model for Sound Event Detection](https://arxiv.org/pdf/2111.15222.pdf), Zhirong Ye and Xiangdong Wang and Hong Liu and Yueliang Qian and Rui Tao and Long Yan and Kazushige Ouchi.
- [SP-SEDT: Self-supervised Pre-training for Sound Event Detection Transformer](https://arxiv.org/pdf/2111.15222.pdf), Zhirong Ye and Xiangdong Wang and Hong Liu and Yueliang Qian and Rui Tao and Long Yan and Kazushige Ouchi.
- [SOUND EVENT DETECTION USING METRIC LEARNING AND FOCAL LOSS FOR DCASE 2021 TASK 4](https://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Tian_130_t4.pdf), Gangyi Tian and Yuxin Huang and Zhirong Ye and Shuo Ma and Xiangdong Wang and Hong Liu and Yueliang Qian and Rui Tao and Long Yan and Kazushige Ouchi and Janek Ebbers and Reinhold Haeb-Umbach.
- [FORWARD-BACKWARD CONVOLUTIONAL RECURRENT NEURAL NETWORKS AND
TAG-CONDITIONED CONVOLUTIONAL NEURAL NETWORKS FOR
WEAKLY LABELED SEMI-SUPERVISED SOUND EVENT DETECTION](https://dcase.community/documents/workshop2020/proceedings/DCASE2020Workshop_Ebbers_69.pdf), Janek Ebbers and Reinhold Haeb-Umbach.

## Citation
```
@techreport{Li2022d,
    author = "Li, Yiming and Guo, Zhifang and Ye, Zhirong and Wang, Xiangdong and Liu, Hong and Qian, Yueliang and Tao, Rui and Yan, Long and Ouchi, Kazushige",
    title = "A Hybrid System Of Sound Event Detection Transformer And Frame-Wise Model For Dcase 2022 Task 4",
    institution = "DCASE2022 Challenge",
    year = "2022",
    month = "June"
}
@article{
  author = "Zhirong, Ye and Xiangdong, Wang and Hong, Liu and Yueliang, Qian and Rui, Tao and Long, Yan and Kazushige, Ouchi",
  title = "Sound Event Detection Transformer: An Event-based End-to-End Model for Sound Event Detection",
  year = "2021",
  url = "https://arxiv.org/abs/2110.02011",
}
@article{
  author = "Zhirong, Ye and Xiangdong, Wang and Hong, Liu and Yueliang, Qian and Rui, Tao and Long, Yan and Kazushige, Ouchi",
  title = "SP-SEDT: Self-supervised Pre-training for Sound Event Detection Transformer",
  year = "2021",
  url = "https://arxiv.org/abs/2111.15222",
}
@techreport{Tian2021,
    author = "Tian, Gangyi and Huang, Yuxin and Ye, Zhirong and Ma, Shuo and Wang, Xiangdong and Liu, Hong and Qian, Yueliang and Tao, Rui and Yan, Long and Ouchi, Kazushige and Ebbers, Janek Haeb-Umbach, Reinhold",
    title = "Sound Event Detection Using Metric Learning And Focal Loss For DCASE 2021 Task 4",
    institution = "DCASE2021 Challenge",
    year = "2021",
    month = "June",
}
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

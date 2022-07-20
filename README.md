<div align="center">
<h1>SpA-Former:Transformer image shadow detection and removal via spatial attention  </h1>
</div>

<div align="center">
<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/zhangbaijin/Spatial-Transformer-shadow-removal?color=green"> <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/zhangbaijin/Spatial-Transformer-shadow-removal">  <img alt="GitHub issues" src="https://img.shields.io/github/issues/zhangbaijin/Spatial-Transformer-shadow-removal"> 
</div>
<div align="center">
<img alt="GitHub watchers" src="https://img.shields.io/github/watchers/zhangbaijin/Spatial-Transformer-shadow-removal?style=social"> <img alt="GitHub stars" src="https://img.shields.io/github/stars/zhangbaijin/Spatial-Transformer-shadow-removal"> <img alt="GitHub forks" src="https://img.shields.io/github/forks/zhangbaijin/Spatial-Transformer-shadow-removal?style=social">
</div>


### new

- 2022.6.30  The draft is released now at [http://arxiv.org/abs/2206.10910](https://arxiv-export1.library.cornell.edu/pdf/2206.10910v1)
SpA-Former:Transformer image shadow detection and removal via spatial attention  
## Results of shadow removal on ISTD dataset

![image](https://github.com/zhangbaijin/SpA-Former-shadow-removal/blob/main/imgs/structure.png))
# Qucikly run
## 1. TRAIN

Modify the `config.yml` to set your parameters and run:

```bash
python train.py
```

## 2. TEST

First，the dataset is trained on 640x480, so you should resize test dataset to 640X480, you can use the code to resize your image 
```bash python bigresize.py```
and then follow the code to test the results:
```bash
python predict.py --config <path_to_config.yml_in_the_out_dir> --test_dir <path_to_a_directory_stored_test_data> --out_dir <path_to_an_output_directory> --pretrained <path_to_a_pretrained_model> --cuda
```
Attention visual results is bellow:[Attention visual results](https://drive.google.com/file/d/188MbZxi3rVB41vAzLX2dssW4sRLYLqyn/view?usp=sharing)
There're my pre-trained models on [ISTD](./pretrained_models/RICE1/)(`./pretrained_models/ISTD/gen_model_epoch_200.pth`) 

Some results are shown as bellow and the images from left to right are: input, attention map, SpA-Former's output, ground truth.

![image](https://github.com/zhangbaijin/SpA-Former-shadow-removal/blob/main/imgs/introduction.png)

## 3. Pretrained model

Download the pretrained model shadow-removal  [Google-drive](https://drive.google.com/drive/folders/1pxwwAfwnGKkLj-GAlkVCevbEQM4basgR?usp=sharing)
 and [Baidu Drive](https://pan.baidu.com/s/1slny1G_9WuxBcoyw5eKUVA)  提取码：rpis
## 4.Test results
Our test results:  [Google-drive](https://drive.google.com/file/d/1m-zE9wxiEL8lO8pX5n65cbi0GQaAGSPr/view?usp=sharing)
and [Baidu drive](https://pan.baidu.com/s/1ek9qaowfPg4CkDaZF6KTCQ)  提取码：18ut

## 5.Evaluate 
To reproduce PSNR/SSIM/RMSE scores of the paper, run MATLAB script
```
evaluate.m
```
In this section, I compares SpA-Former with several methods using peak signal to noise ratio (***PSNR***) and structural similarity index (***SSIM***)  and (***RMSE***) as metrics on datasets ISTD.

![image](https://github.com/zhangbaijin/Spatial-Transformer-shadow-removal/blob/main/compare.jpg))

# ACKNOLAGEMENT
The code is updated on [https://github.com/Penn000/SpA-GAN_for_cloud_removal)]

# 2. DATASET

## 2.1. ISTD_DATASET

Click [official address]([here](https://github.com/nhchiu/Shadow-Removal-ISTD)) Build the file structure as the folder `data` shown. Here `input` is the folder where the shadow image is stored and the folder `target` stores the corresponding no shadow images.

```
./
+-- data
    +--	ISTD_DATASET
        +-- train
        |   +-- input
        |   |   +-- 0.png
        |   |   +-- ...
        |   +-- target
        |       +-- 0.png
        |       +-- ...
        +-- test
            +-- input
            |   +-- 0.png
            |   +-- ...
            +-- target
                +-- 0.png
                +-- ...
```


##  CONTACT

Contact me if you have any questions about the code and its execution.

E-mail: framebreak_zxf@163.com

If you think this work is helpful for your research, give me a star :-D

### Citations
```
@article{zhang2022spa,
  title={SpA-Former: Transformer image shadow detection and removal via spatial attention},
  author={Zhang, Xiao Feng and Gu, Chao Chen and Zhu, Shan Ying},
  journal={arXiv e-prints},
  pages={arXiv--2206},
  year={2022}
```




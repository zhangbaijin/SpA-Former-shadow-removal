<div align="center">
<h1>SpA-Former:Transformer image shadow detection and removal via spatial attention  </h1>
</div>

<div align="center">
<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/zhangbaijin/Spatial-Transformer-shadow-removal?color=green"> <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/zhangbaijin/Spatial-Transformer-shadow-removal">  <img alt="GitHub issues" src="https://img.shields.io/github/issues/zhangbaijin/Spatial-Transformer-shadow-removal"> <img alt="GitHub closed issues" src="https://img.shields.io/github/issues-closed/Penn000/SpA-GAN_for_cloud_removal?color=red">
</div>
<div align="center">
<img alt="GitHub watchers" src="https://img.shields.io/github/watchers/zhangbaijin/Spatial-Transformer-shadow-removal?style=social"> <img alt="GitHub stars" src="https://img.shields.io/github/stars/zhangbaijin/Spatial-Transformer-shadow-removal"> <img alt="GitHub forks" src="https://img.shields.io/github/forks/zhangbaijin/Spatial-Transformer-shadow-removal?style=social">
</div>


### new

- 2022.6.30  The draft is released now at https://arxiv.org/abs/2009.13015.

shadow removal and cloud removal based on CLNet， paper is coming soon
## Results of shadow removal on ISTD dataset

![image](https://github.com/zhangbaijin/Spatial-Transformer-shadow-removal/blob/main/result.png))

## Quick Run

To test the pre-trained models of [Decloud](https://drive.google.com/drive/folders/1hJQVQopWMD0WazeQzZC2eDbtirXkGILO?usp=sharing) on your own images, run 
```
python demo.py --task Task_Name --input_dir path_to_images --result_dir save_images_here
```

## Pretrained model


1. Download the pretrained model [cloud-removal](https://drive.google.com/drive/folders/1hJQVQopWMD0WazeQzZC2eDbtirXkGILO?usp=sharing)

2.Baidu Drive: 链接：https://pan.baidu.com/s/1nBNEsRLIFS2VVtHl8O14Rw 提取码：5mli

# Dataset 
Download datasets RICE from [here](https://github.com/BUPTLdy/RICE_DATASET), and ISTD dataset from [here](https://github.com/nhchiu/Shadow-Removal-ISTD)

#### To reproduce PSNR/SSIM/RMSE scores of the paper, run MATLAB script
```
evaluate.m
```
# ACKNOLAGEMENT
The code is updated on [https://github.com/Penn000/SpA-GAN_for_cloud_removal)]

## 2. DATASET

### 2.1. ISTD_DATASET

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

## 3. TRAIN

Modify the `config.yml` to set your parameters and run:

```bash
python train.py
```

## 4. TEST

```bash
python predict.py --config <path_to_config.yml_in_the_out_dir> --test_dir <path_to_a_directory_stored_test_data> --out_dir <path_to_an_output_directory> --pretrained <path_to_a_pretrained_model> --cuda
```

There're my pre-trained models on [ISTD](./pretrained_models/RICE1/)(`./pretrained_models/RICE1/gen_model_epoch_200.pth`) and [RICE2]((./pretrained_models/RICE1/))(`./pretrained_models/RICE2/gen_model_epoch_200.pth`).

Some results are shown as bellow and the images from left to right are: input, attention map, SpA-Former's output, ground truth.

![image](https://github.com/zhangbaijin/Spatial-Transformer-shadow-removal/blob/main/106-2.png))

## 5. EXPERIMENTS

In this section, I compares SpA-Former with several methods using peak signal to noise ratio (***PSNR***) and structural similarity index (***SSIM***)  and RMSE as metrics on datasets ISTD.

![image](https://github.com/zhangbaijin/Spatial-Transformer-shadow-removal/blob/main/compare.jpg))

## 6. CONTACT

Contact me if you have any questions about the code and its execution.

E-mail:SemiZxf2163.com

If you think this work is helpful for your research, give me a star :-D

### Citations






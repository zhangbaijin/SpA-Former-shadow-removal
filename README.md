<div align="center">
<h1>SpA GAN for Cloud Removal</h1>
</div>

<div align="center">
<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/Penn000/SpA-GAN_for_cloud_removal?color=green"> <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/Penn000/SpA-GAN_for_cloud_removal">  <img alt="GitHub issues" src="https://img.shields.io/github/issues/Penn000/SpA-GAN_for_cloud_removal"> <img alt="GitHub closed issues" src="https://img.shields.io/github/issues-closed/Penn000/SpA-GAN_for_cloud_removal?color=red">
</div>
<div align="center">
<img alt="GitHub watchers" src="https://img.shields.io/github/watchers/Penn000/SpA-GAN_for_cloud_removal?style=social"> <img alt="GitHub stars" src="https://img.shields.io/github/stars/Penn000/SpA-GAN_for_cloud_removal?style=social"> <img alt="GitHub forks" src="https://img.shields.io/github/forks/Penn000/SpA-GAN_for_cloud_removal?style=social">
</div>


### new

- 2020.9.29  The draft is released now at https://arxiv.org/abs/2009.13015.


## 1. INTRODUCTION

This is the source code of [***Cloud Removal for Remote Sensing Imagery via Spatial Attention Generative Adversarial Network***](https://arxiv.org/abs/2009.13015). In this work, I proposes a novel cloud removal model called ***spatial attention generative adversarial networks*** or ***SpA GAN***, which use [spatial attention networks (SPANet)](https://github.com/stevewongv/SPANet) as generator. The architecture of *SpA GAN* is shown as fellow:

- **Generator**

*SpA GAN* uses *spatial attention networks* an generator. See `./models/gen/SPANet.py` for more details.

<div align="center"><img src="./readme_images/SPANet.jpg"></div>

- **Discriminator**

Discriminator is a fully  CNN that **C** is convolution layer, **B** is batch normalization and **R** is Leaky ReLU. See `./models/dis/dis.py` for more details.

<div align="center"><img src="./readme_images/dis.jpg"></div>

- **Loss**

The total loss of *SpA GAN* is formulated as fellow:

<div align="center"><img src="./readme_images/loss_spagan.png"></div>

the first part is the loss of GAN

<div align="center"><img src="./readme_images/loss_cgan.png"></div>

the second part is standard $L_1$ loss where $\lambda_c$ is a hyper parameter to control the weight of each channel to the loss.

<div align="center"><img src="./readme_images/loss_l1.png"></div>

the third part is attention loss where $A$ is the attention map and $M$ is the mask of cloud that computed from $M=|I_{in}-I_{gt}|_1$.

<div align="center"><img src="./readme_images/loss_att.png"></div>

## 2. DATASET

### 2.1. RICE_DATASET

Click [official address](https://github.com/BUPTLdy/RICE_DATASET) or [Google Drive](https://drive.google.com/file/d/1Tsm9qEugNyDKLe4bu06e-2IqEhENu64D/view?usp=sharing) to download the open source RICE dataset. Build the file structure as the folder `data` shown. Here `cloudy_image` is the folder where the cloudy image is stored and the folder `ground_truth` stores the corresponding cloudless images.

```
./
+-- data
    +--	RICE_DATASET
        +-- RICE1
        |   +-- cloudy_image
        |   |   +-- 0.png
        |   |   +-- ...
        |   +-- ground_truth
        |       +-- 0.png
        |       +-- ...
        +-- RICE2
            +-- cloudy_image
            |   +-- 0.png
            |   +-- ...
            +-- ground_truth
                +-- 0.png
                +-- ...
```

### 2.2. Perlin Dataset

Construct the dataset by adding Perlin noise as cloud into the image.

## 3. TRAIN

Modify the `config.yml` to set your parameters and run:

```bash
python train.py
```

## 4. TEST

```bash
python predict.py --config <path_to_config.yml_in_the_out_dir> --test_dir <path_to_a_directory_stored_test_data> --out_dir <path_to_an_output_directory> --pretrained <path_to_a_pretrained_model> --cuda
```

There're my pre-trained models on [RICE1](./pretrained_models/RICE1/)(`./pretrained_models/RICE1/gen_model_epoch_200.pth`) and [RICE2]((./pretrained_models/RICE1/))(`./pretrained_models/RICE2/gen_model_epoch_200.pth`).

Some results are shown as bellow and the images from left to right are: cloudy image, attention map, SpA GAN's output, ground truth.

<div align="center"><img src="./readme_images/test_0000.png"></div>

<div align="center"><img src="./readme_images/test_0026.png"></div>

## 5. EXPERIMENTS

In this section, I compares *SpA GAN* with *conditional GAN* and *cycle GAN* using peak signal to noise ratio (***PSNR***) and structural similarity index (***SSIM***) as metrics on datasets RICE1 and RICE2.

### 5.1 RICE1

**qualitative analysis**

The result are shown as bellow and the images from left to right are: cloudy image, conditional GAN's output, cycle GAN's output , SpA GAN's output, ground truth.

<div align="center"><img src="./readme_images/rice1_result.png"></div>

**quantitative analysis**

|               |  PSNR  | SSIM  |
| :-----------: | :----: | :---: |
|   **cGAN**    | 26.547 | 0.903 |
| **cycle GAN** | 25.880 | 0.893 |
|  **SpA GAN**  | 30.232 | 0.954 |

### 5.1 RICE2

**qualitative analysis**

The result are shown as bellow and the images from left to right are: cloudy image, conditional GAN's output, cycle GAN's output , SpA GAN's output, ground truth.

<div align="center"><img src="./readme_images/rice2_result.png"></div>

**quantitative analysis**

|               |  PSNR  | SSIM  |
| :-----------: | :----: | :---: |
|   **cGAN**    | 25.384 | 0.811 |
| **cycle GAN** | 23.910 | 0.793 |
|  **SpA GAN**  | 28.368 | 0.906 |

## 6. CONTACT

Contact me if you have any questions about the code and its execution.

E-mail: penn000@foxmail.com

If you think this work is helpful for your research, give me a star :-D

### Citations

```
@article{Pan2020,
  title   = {Cloud Removal for Remote Sensing Imagery via Spatial Attention Generative Adversarial Network},
  author  = {Heng Pan},
  journal = {arXiv preprint arXiv:2009.13015},
  year    = {2020}
}
```




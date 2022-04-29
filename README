## GANSim-3D for conditional geomodelling: theory and field application
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![TensorFlow 1.12](https://img.shields.io/badge/tensorflow-1.12-green.svg?style=plastic)
![cuDNN 7.4.1](https://img.shields.io/badge/cudnn-7.4.1-green.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

![Teaser image](./Workflow of field application of GANSim.png) 
**Picture:** *Workflow of field cave reservoir geomodelling using GANSim*

This repository contains the official TensorFlow implementation of the following paper:

> **GANSim-3D for conditional geomodelling: theory and field application**<br>
> Suihong Song (PengCheng Lab, Stanford, and CUPB), Tapan Mukerji (Stanford), Jiagen Hou (CUPB), Dongxiao Zhang (PengCheng Lab), Xinrui Lyu (Sinopec) <br>
> CUPB: China University of Petroleum - Beijing

> Available at my ResearchGate personal profile (https://www.researchgate.net/profile/Suihong-Song)
>
> **Abstract:** We present a Generative Adversarial Networks (GANs)-based 3D reservoir simulation framework, GANSim-3D, where the generator is progressively trained to capture geological patterns and relationships between various input conditioning data and output earth models and is thus able to directly produce multiple 3D realistic and conditional earth models from given conditioning data. Conditioning data can include 3D sparse well facies data, probability maps, and global features like facies proportion. The generator only includes 3D convolutional layers, and once trained on a dataset consisting of small-size data cubes, it can be used for geomodelling of 3D reservoirs of large arbitrary sizes by simply extending the inputs. To illustrate how GANSim-3D is practically used and to verify GANSim-3D, a field karst cave reservoir in Tahe area of China is used as an example. The 3D well facies data and 3D probability map of caves obtained from geophysical interpretation are taken as conditioning data. First, we create training, validation, and test datasets consisting of 64×64×64-size 3D cave facies models integrating field geological patterns, 3D well facies data, and 3D probability maps. Then, the 3D generator is trained and evaluated with various metrics. Next, we apply the pretrained generator for conditional geomodelling of two field cave reservoirs of size 64×64×64 and 336×256×96. The produced reservoir realizations prove to be diverse, consistent with the field geological patterns and the field conditioning data, and robust to noise in the 3D probability maps. Each realization with 336×256×96 cells only takes 0.988 seconds using 1 GPU. 

This study is based on our previous studies presented in my github profile(
GeoModeling_Unconditional_ProGAN, GeoModeling_GANSim-2D_Condition_to_Well_Facies_and_Global_Features, GeoModeling_GANSim-2D_Condition_to_Probability_Maps_and_Others)

For any question, please contact [songsuihong@126.com]<br>


## Resources

Material related to our paper is available via the following links:

- Paper: (my ResearchGate profile) https://www.researchgate.net/profile/Suihong_Song.
- Code: (Github) this repository 
- Training and test datasets: xxxx................
- Pre-trained GANs: xxxxxxxxxxxxxx

## Licenses

All material, including our training dataset, is made available under MIT license. You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.

## System requirements

* Both Linux and Windows are supported, but Linux is suggested.
* 64-bit Python 3.6 installation. We recommend Anaconda3 with numpy 1.14.3 or newer.
* TensorFlow 1.10.0 or newer with GPU support.
* One or more high-end NVIDIA GPUs.
* Codes are not compatible with A100 GPUs currently. 
* NVIDIA driver 391.35 or newer, CUDA toolkit 9.0 or newer, cuDNN 7.4.1 or newer.


## Training dataset

The training dataset (Zenodo, https://zenodo.org/record/3993791#.X1FQuMhKhaR) includes synthesized facies models, corresponding global features, and sparsely distributed well facies data. Training facies models are stored as multi-resolution TFRecords. Each original facies model (64x64) is downsampled into multiple resolutions (32x32, …, 4x4) and stored in `1r*.tfrecords` files for efficient streaming during training. There is a separate `1r*.tfrecords` file for each resolution. Training global features are stored as `*.labels`, training probability maps are stored in `2probimages.tfrecordsand` although they are not used as conditioning data, and training well facies data is stored as `*3wellfacies.tfrecords`. 


### How to make training data as TFRecords?

(1) In our study, we synthesize training facies models using object-based method in Petrel software, and export them into one file as model properties with `"Gslib"` format. An Gslib format example of the exported file is [Format_example_of_simulated_facies_models_from_Petrel.txt](./Code/Format_example_of_simulated_facies_models_from_Petrel.txt).

First lines of the exported file are like:

>PETREL: Properties
>
>17820 % Number of synthesized facies models
>
>Facies unit1 scale1
>
>Facies unit1 scale1
>
>...
>
>Facies unit1 scale1
>
>% Totally, there are 64x64 lines, corresponding to 64x64 pixels in each facies model; each line has 17820 numbers splitted by space, corresponding to 17820 facies code values of 17820 generated facies realizations at each pixel. 0-background mud faceis, 1-channel sand facies, 2-channel bank facies.
>
>0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 ... 0.000000 1.000000 2.000000
>
>0.000000 1.000000 0.000000 0.000000 0.000000 0.000000 ... 0.000000 0.000000 0.000000
>
>...
>
>0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 ... 0.000000 0.000000 0.000000


(2) This exported file containing synthesized facies models is read in [Preparing_training_and_test_datasets.ipynb](./Code/Preparing_training_and_test_datasets.ipynb). The data in the file is rearranged into `(FaciesModelNumber, 1, 64, 64)`. 

In our study, when synthesizing facies models in Petrel, we only consider orientation of channels varying from 0 to 90 degrees, thus in [Preparing_training_and_test_datasets.ipynb](./Code/Preparing_training_and_test_datasets.ipynb), we further enlarge the facies model dataset by reversing the synthesized facies mdoels vertically whose orientation become from -90 to 0 degrees:
```
allimgs = np.concatenate((partimgs, partimgs[::-1,:,:]),2)
```
Other software, like SGeMS, can also be used to simulate the training facies models, as long as the final generated facies models are arranged into `(FaciesModelNumber, 1, 64, 64)`.

Global features (also called labels) are arranged into `(FaciesModelNumber, GlobalFeaturesNumber)`.

(3) The facies models are then used to simulate probability maps in `3 Generate probability maps` of [Preparing_training_and_test_datasets.ipynb](./Code/Preparing_training_and_test_datasets.ipynb). The probability maps are then used to produce well facies data in `4 Generate well facies` of [Preparing_training_and_test_datasets.ipynb](./Code/Preparing_training_and_test_datasets.ipynb). Although probability maps are not used in this case, they should be contained in a TFrecord training data `2probimages.tfrecords`, because `dataset.py` will need to take `2probimages.tfrecords` as inputs.

(4) When downsampling training facies models, two methods were proposed currently: averaging facies codes, or remaining the most frequent facies code. In this paper, here we use the averaging facies codes. In the near future, we will propose to use a third downsampling method: averaging indicator of each facies. 

```
# used to produce low-D with most frequent facies code
#real_img_t = np.expand_dims(real_img, axis = 3)
#real_img_t_c = np.concatenate((real_img_t[:, 0::2, 0::2], real_img_t[:, 0::2, 1::2], real_img_t[:, 1::2, 0::2], real_img_t[:, 1::2, 1::2]), axis = 3)                
#mode, _ = stats.mode(real_img_t_c, axis = 3)
#real_img = np.squeeze(mode, axis = 3)
                
# used to produce low-D with averaging method
real_img = (real_img[:, 0::2, 0::2] + real_img[:, 0::2, 1::2] + real_img[:, 1::2, 0::2] + real_img[:, 1::2, 1::2]) * 0.25  
```

## Training networks

Once the training dataset and related codes are downloaded, you can train your own facies model generators as follows:

1. Edit [config.py](./Code/0_only_conditioning_to_global_features/config.py) or [config.py](./Code/1_conditioning_to_well_facies_alone_or_with_global_features/config.py) to set path `data_dir` (this path points to the folder containing `TrainingData` and `TestData` folders) for the downloaded training data and path for expected results `result_dir`, gpu number. Global feature types are set with following code:
```
labeltypes = [1]  # can include: 0 for 'channelorientation', 1 for 'mudproportion', 2 for 'channelwidth', 3 for 'channelsinuosity'; but the loss for channel orientation has not been designed in loss.py.
# [] for no label conditioning.
```
If using conventional GAN training process (non-progressive training), uncomment the line of code: 
```
#desc += '-nogrowing'; sched.lod_initial_resolution = 64; train.total_kimg = 10000
```
Set if the input well facies data is enlarged (each well facies data occupies 4x4 pixels) or unenlarged (each well facies data only occupies 1x1 pixel), by uncommenting or commenting following line of code:
```
dataset.well_enlarge = True; desc += '-Enlarg';  # uncomment this line to let the dataset output enlarged well facies data; comment to make it unenlarged.
```

2. Edit [train.py](./Code/0_only_conditioning_to_global_features/train.py) or [train.py](./Code/1_conditioning_to_well_facies_alone_or_with_global_features/train.py) to set detailed parameters of training, such as parameters in `class TrainingSchedule` and `def train_progressive_gan`.

3. Set default path as the directory path of downloaded code files, and run the training script with `python train.py`. Or, edit path in [RunCode.py](./Code/0_only_conditioning_to_global_features/RunCode.ipynb) or [RunCode.py](./Code/1_conditioning_to_well_facies_alone_or_with_global_features/RunCode.ipynb), and run `% run train.py` in `RunCode.py` files with Jupyter notebook.

## Assessment of the trained generator

Each of the four pre-trained generators are evaluated using Test dataset (Zenodo, https://zenodo.org/record/3993791#.X1FQuMhKhaR) in `Analyses_of_Trained_Generator-xxxx.ipynb ` files:

(1) for generator only conditioned to global features [Analyses_of_Trained_Generator.ipynb](./Code/0_only_conditioning_to_global_features/Analyses_of_Trained_Generator.ipynb); 

(2) for generator only conditioned to well facies data [Analyses_of_Trained_Generator-WellCond-AfterEnlarg.ipynb](./Code/1_conditioning_to_well_facies_alone_or_with_global_features/Analyses_of_Trained_Generator-WellCond-AfterEnlarg.ipynb); 

(3) for generator conditioned to channel sinuosity and well facies data [Analyses_of_Trained_Generator-Sinuosity-WellEnlarg.ipynb](./Code/1_conditioning_to_well_facies_alone_or_with_global_features/Analyses_of_Trained_Generator-Sinuosity-WellEnlarg.ipynb);

(4) for generator conditioned to mud proportion and well facies data [Analyses_of_Trained_Generator-MudProp-WellEnlarg.ipynb](./Code/1_conditioning_to_well_facies_alone_or_with_global_features/Analyses_of_Trained_Generator-MudProp-WellEnlarg.ipynb).

Detailed steps are illustrated inside these `*.ipynb` files. How to run them is also explained in previous section ` Using pre-trained networks `.

Please note that the exact results may vary from run to run due to the non-deterministic nature of TensorFlow.

## Acknowledgements

Code for this project is improved from the original code of Progressive GANs (https://github.com/tkarras/progressive_growing_of_gans). We thank the authors for their great job.


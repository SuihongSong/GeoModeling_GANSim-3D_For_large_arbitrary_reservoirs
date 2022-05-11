## GANSim-3D for conditional geomodelling: theory and field application
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![TensorFlow 1.12](https://img.shields.io/badge/tensorflow-1.12-green.svg?style=plastic)
![cuDNN 7.4.1](https://img.shields.io/badge/cudnn-7.4.1-green.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

![Teaser image](./Workflow_of_field_application_of_GANSim.png) 
**Picture:** *Workflow of field cave reservoir geomodelling using GANSim*

This repository contains the official TensorFlow implementation of the following paper:

> **GANSim-3D for conditional geomodelling: theory and field application**<br>
> Suihong Song (PengCheng Lab, Stanford, and CUPB), Tapan Mukerji (Stanford), Jiagen Hou (CUPB), Dongxiao Zhang (PengCheng Lab), Xinrui Lyu (Sinopec) <br>
> CUPB: China University of Petroleum - Beijing

> Available at my ResearchGate personal profile (https://www.researchgate.net/profile/Suihong-Song)

> **Abstract:** We present a Generative Adversarial Networks (GANs)-based 3D reservoir simulation framework, GANSim-3D, where the generator is progressively trained to capture geological patterns and relationships between various input conditioning data and output earth models and is thus able to directly produce multiple 3D realistic and conditional earth models from given conditioning data. Conditioning data can include 3D sparse well facies data, probability maps, and global features like facies proportion. The generator only includes 3D convolutional layers, and once trained on a dataset consisting of small-size data cubes, it can be used for geomodelling of 3D reservoirs of large arbitrary sizes by simply extending the inputs. To illustrate how GANSim-3D is practically used and to verify GANSim-3D, a field karst cave reservoir in Tahe area of China is used as an example. The 3D well facies data and 3D probability map of caves obtained from geophysical interpretation are taken as conditioning data. First, we create training, validation, and test datasets consisting of 64×64×64-size 3D cave facies models integrating field geological patterns, 3D well facies data, and 3D probability maps. Then, the 3D generator is trained and evaluated with various metrics. Next, we apply the pretrained generator for conditional geomodelling of two field cave reservoirs of size 64×64×64 and 336×256×96. The produced reservoir realizations prove to be diverse, consistent with the field geological patterns and the field conditioning data, and robust to noise in the 3D probability maps. Each realization with 336×256×96 cells only takes 0.988 seconds using 1 GPU. 

This study is based on our previous studies presented in my github profile(
GeoModeling_Unconditional_ProGAN, GeoModeling_GANSim-2D_Condition_to_Well_Facies_and_Global_Features, GeoModeling_GANSim-2D_Condition_to_Probability_Maps_and_Others)

For any question, please contact [songsuihong@126.com]<br>


## Resources

Material related to our paper is available via the following links:

- Paper: (my ResearchGate profile) https://www.researchgate.net/profile/Suihong_Song.
- Code: [Codes](./Codes/) 
- Dataset used to prepare training datase: [CaveHeightDistributionMaps](./CaveHeightDistributionMaps/)
- Pre-trained GANs: [TrainedGenerator](./TrainedGenerator/) 

## Licenses

All material, including our training dataset, is made available under MIT license. You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.

## System requirements

* Both Linux and Windows are supported, but Linux is suggested.
* 64-bit Python 3.6 installation. We recommend Anaconda3 with numpy 1.14.3 or newer.
* TensorFlow 1.10.0 or newer with GPU support.
* One or more high-end NVIDIA GPUs.
* Codes are not compatible with A100 GPUs currently. 
* NVIDIA driver 391.35 or newer, CUDA toolkit 9.0 or newer, cuDNN 7.4.1 or newer.


## 1. Preparing training/test dataset

The training dataset includes 3D synthesized cave facies models, sparse 3D well facies data, and 3D probability maps. Corresponding global features data are also provided in this project although not used as the input of the generator. 

We have synthesized cave distributions using the proposed process-mimicking approach of the paper, but keep them as 2D cave height distribution maps ([CaveHeightDistributionMaps](./CaveHeightDistributionMaps/)). 

These maps are then recovered into 3D cave facies models, from which 3D probability cubes and well facies cubes are further constructed, see [Preparing training and test datasets for karst caves-3D.ipynb](./Codes/Preparing_training_and_test_datasets_for_karst_caves-3D.ipynb/). 

Training facies models are stored as multi-resolution TFRecords. Each original facies model (64x64x64) is downsampled into multiple resolutions (32x32x32, …, 4x4x4) and stored in `1r*.tfrecords` files for efficient streaming during training. There is a separate `1r*.tfrecords` file for each resolution. Training probability maps are stored in `2probimages.tfrecordsand`, and training well facies data is stored as `*3wellfacies.tfrecords`. Label data is stored in `TrainingData-4rxx.labels`, although it is not used as input of the generator in this project currently. 


## 2. Training networks

Once the training dataset are prepared, GANs can be trained following steps:

(1) Edit [config.py](./Codes/config.py) to set path `data_dir` (this path points to the folder containing `TrainingData` and `TestData` folders produced in previous step) containing the training data and path for expected results `result_dir`, gpu number `num_gpus`, batch size `sched.minibatch_dict`, learning rate, and names, etc.

If using conventional GAN training process (non-progressive training), uncomment the line of code: 
```
#desc += '-nogrowing'; sched.lod_initial_resolution = 64; sched.lod_training_kimg = 0; sched.lod_transition_kimg = 0; train.total_kimg = 10000
```

Set if the input well facies data is enlarged (each well facies data occupies 4x4 pixels) or unenlarged (each well facies data only occupies 1x1 pixel), by uncommenting or commenting following line of code:
```
dataset.well_enlarge = True; desc += '-Enlarg';  # uncomment this line to let the dataset output enlarged well facies data; comment to make it unenlarged.
```

(2) Edit [loss.py](./Codes/loss.py) to revise the weights for GANs loss `orig_weight`, well facies-condition loss `Wellfaciesloss_weight`, probability map-condition loss `Probcubeloss_weight`, various global features (labels)-condition loss, batch multiplier used to calculate frequency map (approximating probability map) when training `batch_multiplier`, etc.

(3) Edit [train.py](./Codes/train.py) to set detailed parameters of training, such as parameters in `class TrainingSchedule` and `def train_progressive_gan`.

(4) Run [train.py](./Codes/train.py) with python.

#### The generator we have trained is also available here for direct application, see [network-snapshot-003360.pkl](./TrainedGenerator/)


## 3. Assessment of the trained generator

The pre-trained generators are evaluated using Test dataset (constructed in step 1) in [Evaluations_of_Trained_Generator.ipynb](./Codes/Evaluations_of_Trained_Generator.ipynb). 


## 4. Field reservoir geomodelling using the trained generator

The trained generator is finally used for geomodelling of two field reservoirs with size of 64x64x64 and 96x256x336, where each cell is the same.

### 4.1 Field reservoir with 64x64x64 cells

The field obtained 3D well facies and probability map (interpreted from field seismic data) cubes are availabe at [Field measured 3D well facies and probability maps](./PracticalDataFromTahe/64x64x64/).

See [Field_Application_of_Trained_Generator_for_64x64x64-size.ipynb](./Codes/Field_Application_of_Trained_Generator_for_64x64x64-size.ipynb/) for detailed steps about how to produce facies model realizations by taking the given conditioning data into the trained generator. Note to revise the paths to conditioning data, codes, and produced realizations in the file.

### 4.2 Field reservoir with 96x256x336 cells (a large arbitrary-size reservoir)

The field obtained 3D well facies and probability map (interpreted from field seismic data) cubes are availabe at [Field measured 3D well facies and probability maps](./PracticalDataFromTahe/96x256x336/) with the format of `rar`. Please unpack the `.rar` file first before using the data in next step.

See [Field_Application_of_Trained_Generator_for_Arbitary_Large_Size.py](./Codes/Field_Application_of_Trained_Generator_for_Arbitary_Large_Size.py/) for detailed steps about how to produce facies model realizations by taking the given conditioning data into the trained generator. Note producing such large-size reservoir realizations require lots of GPU memory. The paths to conditioning data, codes, and produced realizations should be revised. Run this `.py` file to produce realizations.


Please give appropriate credit to our work, if it is valuable for you to some extent.

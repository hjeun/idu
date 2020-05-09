# Information Discrimination Units (IDU)
**Learning to Discriminate Information for Online Action Detection, CVPR 2020**  
Hyunjun Eun, Jinyoung Moon, Jongyoul Park, Chanho Jung, Changick Kim  
[[`arXiv`](https://arxiv.org/abs/1912.04461)]

## Introduction
<div align="center">
  <img src="https://drive.google.com/file/d/1Mnkq2Mj0VeQvdEcxd-OQkFRFJ_rtk1E1/view?usp=sharing" width="700px" />
</div>
This is official implementation of IDU. IDU aims to 

## Updates
**30 May, 2020**: Initial update

## Installation

### Prerequisites
- Ubuntu 16.04  
- Python 3.6.8   
- CUDA 10.0  
- cuDNN 7.5

### Requirements
- tensorflow==1.13.1  
- scipy==1.3.0  

## Training
The code for training is not included in this repository, and we cannot release the full training code for

### Trained Models
Download trained models on [THUMOS-14](https://drive.google.com/uc?export=download&confirm=8b-Z&id=1DUpOzbLG-8ptpPVJrzsRiKFRIaO1FBcs) and [TVSeries](https://drive.google.com/uc?export=download&confirm=A33G&id=1zJ7EfhQg91XVrV9ryjvmcoyQDEl3U_Fn) in 'dataset name'/logs/'.

## Testing

__For THUMOS-14__  
```
python thumos14/test_tfn.py --model_name TFN-ED
```
The code provides the results of TFN-ED, '__mAP of 55.7%__'.  
An AP value for each class are also provided, as reported in our supplementary material.

__For TVSeries__  
```
python tvseries/test_tfn.py --model_name TFN-ED
```
The code provides the results of TFN-ED, '__mcAP of 85.0%__'.  
A cAP value for each class are also provided, as reported in our supplementary material.

## Citing IDU
Please cite our paper in your publications if it helps your research:

```BibTeX
@article{eun2020idu,
  title={Learning to Discriminate Information for Online Action Detection},
  author={Eun, Hyunjun and Moon, Jinyoung and Park, Jongyoul and Jung, Chanho and Kim, Changick},
  booktitle={CVPR},
  year={2020}
} 
```

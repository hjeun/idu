# Information Discrimination Units (IDU)
**Learning to Discriminate Information for Online Action Detection, CVPR2020**

Hyunjun Eun, Jinyoung Moon, Jongyoul Park, Chanho Jung, Changick Kim

[[`arXiv`](https://arxiv.org/abs/1912.04461)]

## Description
This is official implementation of IDU. IDU aims to 

## Prerequisites
- Ubuntu 16.04  
- Python 3.6.8   
- CUDA 10.0  
- cuDNN 7.5

## Requirements
- Tensorflow==1.13.1  
- scipy==1.3.0  
```
pip install -r requirements.txt
```

## Trained Models
<a href="https://drive.google.com/uc?export=download&confirm=8b-Z&id=1DUpOzbLG-8ptpPVJrzsRiKFRIaO1FBcs">THUMOS-14</a>  
<a href="https://drive.google.com/uc?export=download&confirm=A33G&id=1zJ7EfhQg91XVrV9ryjvmcoyQDEl3U_Fn">TVSeries</a>  
Model files should be located in 'dataset name'/logs/'.

# Testing
Three architectures (TFN-ED, TFN-NL, and TFN-S) can be tested.  

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

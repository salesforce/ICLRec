# Introduction
Intent Contrastive Learning for Sequential Recommendation (ICLRec)

Source code for paper: [Intent Contrastive Learning for Sequential Recommendation](https://arxiv.org/pdf/2202.02519.pdf)

Motivation: 

Users' interactions with items are driven by various underlying intents. These intents are potentially beneficial to learn a better users' preferences toward massive item set.

<img src="./img/motivatioin_sports.png" width="800">

Model Architecture:

<img src="./img/model.png" width="600">

# Reference

Please cite our paper if you use this code.

```
@article{chen2022intent,
  title={Intent Contrastive Learning for Sequential Recommendation},
  author={Chen, Yongjun and Liu, Zhiwei and Li, Jia and McAuley, Julian and Xiong, Caiming},
  journal={arXiv preprint arXiv:2202.02519},
  year={2022}
}
```

# Implementation
## Requirements

Python >= 3.7  
Pytorch >= 1.2.0  
tqdm == 4.26.0

## Datasets

Four prepared datasets are included in `data` folder.

## Train Model

To train ICLRec on `Sports_and_Outdoors` dataset, change to the `src` folder and run following command: 

```
python main.py --data_name Sports_and_Outdoors
```

The script will automatically train ICLRec and save the best model found in validation set, and then evaluate on test set.


## Evaluate Model

You can directly evaluate a trained model on test set by running:

```
python main.py --data_name Sports_and_Outdoors --model_idx 0 --do_eval
```

We provide a model that trained on Sports_and_Games dataset in `./src/output` folder. Please feel free to test is out.

# Acknowledgment
 - Transformer and training pipeline are implemented based on [S3-Rec](https://github.com/RUCAIBox/CIKM2020-S3Rec). Thanks them for providing efficient implementation.


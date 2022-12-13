# 2022-COMP576-FinalProject
Team member: Mingjing Xu (mx12), Sophie Sun (ys97), Yixuan Liu(yl244), Yufei Wang(yw116)
### 0) Overview
This repository contains improved PyTorch implementations of SRGAN model on Covid-19 CT images. 

### 1) Train:
```
python main.py --data_dir xxx --train_dataset xxx --test_dataset xxx --save_dir xxx  --mode train_and_test
```
### 2) Inference:
```
python main.py --data_dir xxx --test_dataset xxx --save_dir xxx  --mode predict
```

```
--data_dir: Path of test data folder
--test_dataset: Test data folder name
--train_dataset: Training data folder name
--save_dir: Path of saving checkpoints
```

### 3)Reference:
**SRGAN** : [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)


**Comparison with Different Methods**: [Collection of Super-Resolution models Github Page](https://github.com/togheppi/pytorch-super-resolution-model-collection)

This repository contains the code for the paper **"Revisiting Reweighted Risk for Calibration: AURC, Focal, 
and Inverse Focal Loss"** [[arxiv](https://arxiv.org/pdf/2505.23463)].

#### Key Dependencies
To run the code, you will need the following dependencies (excluding common packages like `scipy`, `numpy`, and `torch`):

- **Python** ≥ 3.8

#### Preparing Datasets and Models

- **Tiny-ImageNet [[link](https://github.com/tjmoon0104/pytorch-tiny-imagenet)]**
  
To download and preprocess the dataset, use the following commands:
```bash
cd data
python tiny_imagenet_utils.py
```
- **CIFAR-10/100**

#### Using the select AU loss in your project

To train the model with select AU loss, you can copy the file `loss/select_au.py` into your repository. 

#### Train the model with select AU loss

To train the model with select AU loss, use the following commands:
```bash
python src/train.py --arch vit_small --dataset tiny-imagenet --loss_type select_au --seed 40 --workers 1
```

#### Reference
If you found this work or code useful, please cite:

```
@misc{zhou2025revisitingreweightedriskcalibration,
      title={Revisiting Reweighted Risk for Calibration: AURC, Focal, 
and Inverse Focal Loss}, 
      author={Han Zhou and Sebastian G. Gruber and Teodora Popordanoska 
and Matthew B. Blaschko},
      year={2025},
      eprint={2505.23463},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.23463}, 
}
```
#### License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

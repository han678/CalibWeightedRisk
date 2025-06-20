This repository contains the code for the paper **"Revisiting Reweighted Risk for Calibration: AURC, Focal Loss, 
and Inverse Focal Loss"** [[arxiv](https://arxiv.org/pdf/2505.23463)].

#### Key Dependencies
To run the code, you will need the following dependencies (excluding common packages like `scipy`, `numpy`, and `torch`):

- **Python** ≥ 3.8
- **torchsort**: A library that implements fast, differentiable sorting and ranking in PyTorch. [Learn more here](https://github.com/google-research/fast-soft-sort).

  Install via pip:
  ```bash
  pip install torchsort
  ```

#### Preparing Datasets and Models

- **Tiny-ImageNet [[link](https://github.com/tjmoon0104/pytorch-tiny-imagenet)]**
To download and preprocess the dataset, use the following commands:
```bash
cd data
python tiny_imagenet_utils.py
```
- **CIFAR-10/100**

#### Using the AURC loss in your project

To train the model with regularized AURC loss, you can copy the file `loss/aurc.py` into your repository. 

#### Train the model with AURC loss

To train the model with regularized AURC loss, use the following commands:
```bash
cd src
python main.py
```

#### Reference
If you found this work or code useful, please cite:

```
@misc{zhou2025revisitingreweightedriskcalibration,
      title={Revisiting Reweighted Risk for Calibration: AURC, Focal Loss, 
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

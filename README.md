# CMG-Net: Robust Normal Estimation for Point Clouds via Chamfer Normal Distance and Multi-scale Geometry  (AAAI 2024)

### *[ArXiv](https://arxiv.org/abs/2312.09154) 

This work presents an accurate and robust method for estimating normals from point clouds. In contrast to predecessor approaches that minimize the deviations between the annotated and the predicted normals directly, leading to direction inconsistency, we first propose a new metric termed Chamfer Normal Distance to address this issue. This not only mitigates the challenge but also facilitates network training and substantially enhances the network robustness against noise. Subsequently, we devise an innovative architecture that encompasses Multi-scale Local Feature Aggregation and Hierarchical Geometric Information Fusion. This design empowers the network to capture intricate geometric details more effectively and alleviate the ambiguity in scale selection. Extensive experiments demonstrate that our method achieves the state-of-the-art performance on both synthetic and real-world datasets, particularly in scenarios contaminated by noise. This project is the implementation of CMG-Net by Pytorch.

## Requirements
The code is implemented in the following environment settings:
- Ubuntu 20.04
- CUDA 11.3
- Python 3.8
- Pytorch 1.12
- Numpy 1.24
- Scipy 1.10

## Dataset
We train our network model on the [PCPNet](http://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/pclouds.zip) dataset.
Download the dataset to the folder `***/dataset/` and copy the list into the fold `***/dataset/PCPNet/list`. The dataset is organized as follows:
```
│dataset/
├──PCPNet/
│  ├── list
│      ├── ***.txt
│  ├── ***.xyz
│  ├── ***.normals
│  ├── ***.pidx
```

## Train
Our trained model is provided in `./log/001/ckpts/ckpt_900.pt`.
To train a new model on the PCPNet dataset, simply run:
```
python train.py
```
Your trained model will be save in `./log/***/ckpts/`.

## Test
You can use the provided model for testing:
```
python test.py
```
The evaluation results will be saved in `./log/001/results_PCPNet/ckpt_900/`.

To test with your trained model, you need to change the variables in `test.py`:
```
ckpt_dirs       
ckpt_iter
```
To save the normals of the input point cloud, you need to change the variables in `test.py`:
```
save_pn = True        # to save the point normals as '.normals' file
sparse_patches = False  # to output sparse point normals or not
```

## Acknowledgement
The code is heavily based on [HSurf-Net](https://github.com/LeoQLi/HSurf-Net).
If you find our work useful in your research, please cite the following papers:

```
@inproceedings{wu2024cmg,
  title={CMG-Net: Robust Normal Estimation for Point Clouds via Chamfer Normal Distance and Multi-scale Geometry},
  author={Wu, Yingrui and Zhao, Mingyang and Li, Keqiang and Quan, Weize and Yu, Tianqi and Yang, Jianfeng and Jia, Xiaohong and Yan, Dong-Ming},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  year={2024}
}

@inproceedings{ben2020deepfit,
  title={Deepfit: 3d surface fitting via neural network weighted least squares},
  author={Ben-Shabat, Yizhak and Gould, Stephen},
  booktitle={European conference on computer vision},
  pages={20--34},
  year={2020},
  organization={Springer}
}

@article{li2022hsurf,
  title={HSurf-Net: Normal estimation for 3D point clouds by learning hyper surfaces},
  author={Li, Qing and Liu, Yu-Shen and Cheng, Jin-San and Wang, Cheng and Fang, Yi and Han, Zhizhong},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={4218--4230},
  year={2022}
}
```


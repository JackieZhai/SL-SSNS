# SL-SSNS
Official implementation of ["Addressing Distribution Mismatch for Effective Semi-Supervised Neuron Segmentation"](https://www.biorxiv.org/content/10.1101/2024.05.26.595303v1) [under review]
For convenience, we provide a [demo](https://colab.research.google.com/drive/1vPYYeaycpdQjDiu_TQD4LqQbjezf40yc#scrollTo=zy73yxP8xp2F) illustrating the subvolume selection process in both the spatial and embedding domains, which can be readily applied to your own data.
## Selective Labeling
### Pretraining
```
cd Pretraining
```
```
python pretraining.py
```
### CGS Selection
```
cd CGS
```
```
python CGS.py
```
## Semi-supervised Training
```
cd IIC-Net
```
### Supervised Warm-up
```
python warmup.py
```
### Mixed-view Consistency Regularization
```
python semi_tuning.py
```
## Acknowledgement
This code is based on [SSNS-Net](https://github.com/weih527/SSNS-Net) (IEEE TMI'22) by Huang Wei et al. The postprocessing tools are based on [constantinpape/elf](https://github.com/constantinpape/elf) and [funkey/waterz](https://github.com/funkey/waterz). Should you have any further questions, please let us know. Thanks again for your interest.

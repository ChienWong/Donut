# Donut
##### 此项目为东南大学秋季课程www项目
### This is a tensorflow2.0 realization of Donut model(A anomaly detection model based on VAE for time series) 
### original author's [implementation](https://github.com/NetManAIOps/donut)
## Requirement
### Tensorflow>=2.6 , pandas, and numpy (little api about dataset in tensorflow2.0 is incompatiable, so the requirement is >=2.6)
## Training
### `python train.py`
## Reproduction of some paper data
### server_res_eth1out_curve_6 dataset with 10% anomaly points Manually 
##### blue point is true manual anomaly point and red X is predict anomaly point with best F-score to decide threshlod
![Figure1](https://github.com/ChienWong/Donut/blob/main/figure/Figure_server_res_eth1out_curve_6_0.1.png)
### cpu4 dataset with 1% anomaly points Manually 
![Figure2](https://github.com/ChienWong/Donut/blob/main/figure/Figure_cpu4_0.01.png)
### Best F score with 10% anomaly points Manually
![Figure3](https://github.com/ChienWong/Donut/blob/main/figure/Figure_F_score_0.1.png)
### Best F score with 1% anomaly points Manually
![Figure4](https://github.com/ChienWong/Donut/blob/main/figure/Figure_f_score_0.01.png)
### AUC with 10% anomaly points Manually
![Figure5](https://github.com/ChienWong/Donut/blob/main/figure/Figure_AUC_0.1.png)
### AUC with 1% anomaly points Manually
![Figure6](https://github.com/ChienWong/Donut/blob/main/figure/Figure_auc_0.01.png)
### Z sample from dataset cpu4 and 2 dims
![Figure7](https://github.com/ChienWong/Donut/blob/main/figure/Figure_z_2dims.png)
### Z sample from dataset cpu4 and 3 dims
![Figure8](https://github.com/ChienWong/Donut/blob/main/figure/Figure_z_3d_3.png)
### Best F Score with z_dims
![Figure9](https://github.com/ChienWong/Donut/blob/main/figure/Figure_zdims_F-score.png)
### AUC with z_dims
![Figure10](https://github.com/ChienWong/Donut/blob/main/figure/Figure_zdims_AUC.png)

# SSL-contrastive
Self-supervised contrastive learning to decipher plasma frequency from space-based wave receivers

Codes have been approved for public release; distribution is unlimited. Public Affairs release approval #AFRL-2024-4503

A manuscript titled "Decoding space radio waves: Self-supervised AI decipher plasma frequency" by Su and Carilli details the construction of the software. 

Please refer to https://doi.org/10.1029/2023RS007907 for the loss functions used in the downstream task and the prediction metrics - Huasdoff Distance

Self-supervised pretext tasks:
  * BYOL - main program train_byol.py with input parameters specified in config_byol.yaml
  * PIXCL - main program train_picxl.py with input parameters specified in config_pixcl.yaml

Downstream task:
  * FCN - main program downstream_fcn.py with input parameters specified in config_fcn.yaml

Prediction: main program predict_fcn.py with input parameters specified in config_predict.yaml

Other python source codes are located in the "utils" folder
  * data_loader.py, custom_transform.py, pixcl_multi.py are called by train_byol.py and train_pixcl.py
  * data_loader_downstream.py, pixcl_multi.py, fcn.py, hausdorff.py and dice score.py are called by downstream_fcn.py
  * data_loader_downstream.py, pixcl_multi.py, fcn.py, and hausdorff.py are called by predict_fcn.py

The software utilized in this project was obtained, modified, and consolidated from the following sources. Our version was meticulously undertaken to ensure precise alignment with the specific requirements and objectives of our investigation.

BYOL - Bootstrap Your Own Latent 
       
       original architecture described in https://arxiv.org/abs/2006.07733
       sample software https://github.com/lucidrains/byol-pytorch

PIXCL - PIXel-level Consistency Learning 
        
        original architecture described in https://arxiv.org/abs/2011.10043
        sample software https://github.com/lucidrains/pixel-level-contrastive-learning

FCN  - Fully Convolutional Networks
       
       original architecture described in https://arxiv.org/abs/1411.4038
       easiest FCN https://github.com/pochih/FCN-pytorch

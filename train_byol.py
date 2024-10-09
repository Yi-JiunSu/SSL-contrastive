"""
  developed by Yi-Jiun Su
  The latest update in April 2024
"""
import yaml
import torch
import torchvision.transforms as T
import utils.custom_transform as CT
from utils.data_loader import CustomDataset
import torchvision 
from utils.pixcl_multi import NetWrapperMultiLayers, PPM, MLP, BYOLTrainer 
import os
from copy import deepcopy
#from utils.adam_hd import Adam_HD

print(torch.__version__)
# controlling sources of randomness
torch.manual_seed(0)

if __name__ == '__main__':
    config = yaml.safe_load(open("config_byol.yaml", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Learning with: {device}")

    # define data Transform functions
    # T1 = T.Compose([CT.Jitter(),CT.Scale()])
    T1 = T.Compose([CT.Jitter(),CT.FFT()])
    T2 = T.Compose([CT.WrapHW(dim=2),CT.WrapHW(dim=3)])
    dataset = CustomDataset(**config['dataset'])

    if config['backbone_arch'] == 'resnet18':
        resnet = torchvision.models.resnet18()
    elif config['backbone_arch'] == 'resnet34':
        resnet = torchvision.models.resnet34()
    elif config['backbone_arch'] == 'resnet50':
        resnet = torchvision.models.resnet50()

    #online_encoder include instance- (MLP) & pixel-level (ConvMLP) projectors
    online_encoder = NetWrapperMultiLayers(net=resnet).to(device)
# Not working in a virtural environment yet
#    if torch.cuda.device_count() > 1:
#        print("Let's use", torch.cuda.device_count(), "GPUs!")
#        online_encoder = torch.nn.DataParallel(online_encoder)

    # instance level predictor
    online_predictor = MLP(
        config['learner']['projection_size'],
        config['learner']['projection_size'],
        config['learner']['projection_hidden_size']).to(device)
    
    if config['opt_method'] == 'adam_hd':
        opt = Adam_HD(list(online_encoder.parameters())+list(online_predictor.parameters()), 
              **config['optimizer'])
    elif config['opt_method'] == 'radam':
        opt = torch.optim.RAdam(list(online_encoder.parameters())+list(online_predictor.parameters()),
              **config['optimizer'])
    else: 
        opt = torch.optim.Adam(list(online_encoder.parameters())+list(online_predictor.parameters()),
              **config['optimizer'])

    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,**config['lr_sch_param'])

    if config['pretrain_dir']:
        try:
            checkpoints_dir = os.path.join('./runs',config['pretrain_dir'],'checkpoints')
            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(
                checkpoints_dir,'model_epoch'+config['pretrain_epoch']+'.pth')),
                map_location=torch.device(device))
            online_encoder.load_state_dict(load_params['online_encoder_state_dict'])
#            online_predictor.load_state_dict(load_params['online_predictor_state_dict'])
            opt.load_state_dict(load_params['optimizer_state_dict'])
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    with torch.no_grad():
        target_encoder = deepcopy(online_encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False

    learner = BYOLTrainer(
        online_encoder=online_encoder, 
        target_encoder=target_encoder, 
        online_predictor=online_predictor, 
        optimizer=opt, 
        scheduler=sch, 
        device=device,
        augment1=T1, 
        augment2=T2,
        **config['learner'])

    learner.train(dataset)

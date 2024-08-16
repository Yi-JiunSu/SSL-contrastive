"""
  developed by Yi-Jiun Su
  the latest update in November 2023
"""
import yaml
import torch
import torchvision
import os
from utils.pixcl_multi import NetWrapperMultiLayers
from utils.fcn import FCN32s, FCN16s, FCN8s, FCNs
from PIL import Image
from utils.data_loader_downstream import CustomDataset
import cv2
from utils.hausdorff import hausdorff_distance
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shutil import copyfile

def predict_img(net, device, pil_img, out_threshold: float=0.5):
    net.eval()
    img = torch.from_numpy(CustomDataset.preprocess(None, pil_img, scale=1., is_mask=False))
    img = img.unsqueeze(0) # add the batch dimension before CHW
    img = img.to(device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img).cpu()
        output = torch.nn.functional.interpolate(output, (pil_img.size[1], pil_img.size[0]), mode='bilinear')
        mask = torch.sigmoid(output) > out_threshold
    return mask[0].long().squeeze().numpy()

def plot_img_and_mask(img, pred_mask, true_mask, name, out_dir, hd):
    org_mask1 = np.transpose(np.nonzero( true_mask > 0))
    mask1 = np.transpose(np.nonzero( pred_mask > 0))
    """
    alternative: use torch
    true_mask = torch.tensor(true_mask)
    pred_mask = torch.tensor(pred_mask)
    org_mask1 = torch.nonzero(  true_mask > 0 )
    mask1 = torch.nonzero( pred_mask > 0 )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    """
    fig = plt.figure(figsize=(18,6))
    ax0 = plt.subplot(1, 3, 1)
    ax0.imshow(img)
    ax0.plot( org_mask1[:,1], org_mask1[:,0],'k.',markersize=3)
    ax0.axis('off')
    ax0.set_title('Image + original mask')
    ax1 = plt.subplot(1, 3, 2)
    ax1.imshow(img)
    ax1.plot( mask1[:,1], mask1[:,0],'k.',markersize=3)
    ax1.axis('off')
    ax1.set_title('Image + predicted mask')
    ax2 = plt.subplot(1, 3, 3)
    ax2.plot(org_mask1[:,1],org_mask.shape[0]-org_mask1[:,0],'k.',label='truth')
    ax2.plot(mask1[:,1],org_mask.shape[0]-mask1[:,0],'r.',label='predicted')
    ax2.set_box_aspect(org_mask.shape[0]/org_mask.shape[1])
    ax2.set_xlim(0, org_mask.shape[1])
    ax2.set_ylim(0, org_mask.shape[0])
    ax2.set_title('Hausdorff Distance(95% conf)={:6.3f}$\pm${:5.3f}'.format(hd['mean'],hd['c95']))
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Frequency step')
    ax2.legend()
    plt.savefig(out_dir+'/testimg_'+name)
    plt.close()
#    plt.show()

if __name__ == '__main__':
    config = yaml.safe_load(open("config_predict.yaml", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['backbone_arch'] == 'resnet18':
        resnet = torchvision.models.resnet18()
    elif config['backbone_arch'] == 'resnet34':
        resnet = torchvision.models.resnet34()
    elif config['backbone_arch'] == 'resnet50':
        resnet = torchvision.models.resnet50()

    backbone = NetWrapperMultiLayers(net=resnet).to(device)
    tmp = torch.autograd.Variable(torch.randn(1,3,
        config['image']['sizeH'],
        config['image']['sizeW']))
    _, c5 = backbone(tmp.to(device))

    if config['downstream_arch'] == 'fcn32s':
        fcn = FCN32s(pretrained_net=backbone, n_class=config['image']['n_class'], in_channels=c5.shape[1])
    elif config['downstream_arch'] == 'fcn16s':
        fcn = FCN16s(pretrained_net=backbone, n_class=config['image']['n_class'], in_channels=c5.shape[1])
    elif config['downstream_arch'] == 'fcn8s':
        fcn = FCN8s(pretrained_net=backbone, n_class=config['image']['n_class'], in_channels=c5.shape[1])
    else:
        fcn = FCNs(pretrained_net=backbone, n_class=config['image']['n_class'], in_channels=c5.shape[1])
    fcn = fcn.to(device)

    if config['pretrain_dir']:
        try:
            checkpoints_dir = os.path.join(config['pretrain_dir'],'checkpoints')
            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(
                checkpoints_dir,'model_epoch'+config['pretrain_epoch']+'.pth')),
                map_location=torch.device(device))
            mask_values = load_params.pop('mask_values', [0, 1])
            fcn.load_state_dict(load_params['fcn_state_dict'])
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    img_dir = config['image']['dir']
    in_files = os.listdir(img_dir)
    out_dir = config['predict']['out_dir']
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    copyfile('config_predict.yaml',out_dir+'/config_predict.yaml')
    copyfile('predict_fcn.py',out_dir+'/predict_fcn.py')
    fout = open(out_dir+'/prediction_metrices.txt', 'w')
    fout.write("Prediction Metrices based on Hausdorff Distance\n")
    fout.write("filename    HD-mean HD-median  HD-std  HD-c95   HD-var  HD-max\n")

    for i, name in enumerate(in_files):
        img = Image.open(img_dir+'/'+name).convert("RGB")
        mask = predict_img(net=fcn, device=device, pil_img=img, out_threshold=config['predict']['out_threshold'])
        mask_name = config['image']['mask_dir']+'/'+os.path.splitext(name)[0]+config['image']['mask_suffix']+'.png'
        org_mask = cv2.imread(mask_name, 0)
        m_values = [org_mask.min(),org_mask.max()]
        true_mask = np.zeros((org_mask.shape[0],org_mask.shape[1]),dtype=np.int64)
        for i, v in enumerate(m_values):
            true_mask[org_mask == v] = len(m_values)-1-i
            # convert 255 to be assigned "0" and 0 to be assigned as "1"
        hd = hausdorff_distance(mask, true_mask)
        fout.write('{0} {1:8.5f} {2:8.5f} {3:8.5f} {4:8.5f} {5:8.5f} {6:8.5f}\n'.format(name,
                hd["mean"],hd["median"],hd["std"],hd["c95"],hd["var"],hd["max"]))
        if config['predict']['plot']:
            plot_img_and_mask(img, mask, true_mask, name, config['predict']['out_dir'], hd)

    fout.close()

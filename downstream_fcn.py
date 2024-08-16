"""
  developed by Yi-Jiun Su
  the latest update in March 2024
"""
import yaml
import torch
import torchvision
from utils.pixcl_multi import NetWrapperMultiLayers
import os
from utils.data_loader_downstream import CustomDataset
from torch.utils.data import DataLoader, random_split
from kornia.losses import BinaryFocalLossWithLogits
from utils.dice_score import focal_tversky_loss
from utils.hausdorff import HausdorffERLoss
from utils.fcn import * 
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
from utils.hausdorff import hausdorff_distance
import logging

def _create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))

@torch.inference_mode()
def evaluate(net, dataloader, device, criterion):
    net.eval()
    num_val_batches = len(dataloader)
    running_lossF = 0
    running_lossT = 0
    running_lossH = 0
    with torch.cuda.amp.autocast():
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation', unit='batch', leave=False):
            for batch in dataloader:
                image, true_mask = batch['image'], batch['mask']
                image = image.to(device=device, dtype=torch.float32)
                true_mask = true_mask.to(device=device, dtype=torch.long)
                pred_mask = net(image)
                assert true_mask.min() >=0 and true_mask.max() <=1, 'True mask indices should be in [0, 1]'
                loss1 = criterion(pred_mask.squeeze(1), true_mask.float())
                loss2 = focal_tversky_loss(torch.sigmoid(pred_mask.squeeze(1)),
                        true_mask.float(), multiclass=False, alpha=0.3, beta=0.7, gamma=2.)
                mask = torch.sigmoid(pred_mask) > 0.5
                hd = hausdorff_distance(mask.squeeze().cpu().numpy(), true_mask.squeeze().cpu().numpy())
                running_lossF += loss1.item()
                running_lossT += loss2.item()
                running_lossH += hd['mean']
    net.train()
    running_lossF = running_lossF/max(num_val_batches, 1)
    running_lossT = running_lossT/max(num_val_batches, 1)
    running_lossH = running_lossH/max(num_val_batches, 1)
    return running_lossF, running_lossT, running_lossH

if __name__ == '__main__':
    config = yaml.safe_load(open("config_fcn.yaml", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training with: {device}")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if config['backbone_arch'] == 'resnet18':
        resnet = torchvision.models.resnet18()
    elif config['backbone_arch'] == 'resnet34':
        resnet = torchvision.models.resnet34()
    elif config['backbone_arch'] == 'resnet50':
        resnet = torchvision.models.resnet50()

    backbone = NetWrapperMultiLayers(net=resnet).to(device)
    tmp = torch.autograd.Variable(torch.randn(
        config['trainer']['batch_size'],
        config['dataset']['channels'],
        config['image']['sizeH'],
        config['image']['sizeW']))
    _, c5 = backbone(tmp.to(device))

    if config['pretrain_dir']:
        try:
            checkpoints_dir = os.path.join(config['pretrain_dir'],'checkpoints')
            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(
                checkpoints_dir,'model_epoch'+config['pretrain_epoch']+'.pth')),
                map_location=torch.device(device))
            backbone.load_state_dict(load_params['online_encoder_state_dict'])
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    if config['downstream_arch'] == 'fcn32s':
        fcn = FCN32s(pretrained_net=backbone, n_class=config['image']['n_class'], in_channels=c5.shape[1])
    elif config['downstream_arch'] == 'fcn16s':
        fcn = FCN16s(pretrained_net=backbone, n_class=config['image']['n_class'], in_channels=c5.shape[1])
    elif config['downstream_arch'] == 'fcn8s':
        fcn = FCN8s(pretrained_net=backbone, n_class=config['image']['n_class'], in_channels=c5.shape[1])
    else:
        fcn = FCNs(pretrained_net=backbone, n_class=config['image']['n_class'], in_channels=c5.shape[1])
    fcn = fcn.to(device)

    # Create Dataset
    dataset = CustomDataset(**config['dataset'])

    #Split dataset into train & validation partitions
    n_val = int(len(dataset)* config['trainer']['val_percent'])
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # Create data loaders
    loader_args = dict(
        batch_size=config['trainer']['batch_size'], 
        num_workers=config['trainer']['num_workers'] if config['trainer']['num_workers'] != 'None' else os.cpu_count(), 
        pin_memory=True)
    train_loader = DataLoader(train_set,shuffle=True,drop_last=False,**loader_args)
    val_loader = DataLoader(val_set,shuffle=False,drop_last=True,**loader_args)

    # Set up Loss Functions, Optimizer
    kwargs = {"alpha":0.6, "gamma":2.0, "reduction":'mean'}
    criterion = BinaryFocalLossWithLogits(**kwargs)
    criterionH = HausdorffERLoss(alpha=2.0,k=10)
    optimizer = torch.optim.Adam(fcn.parameters(), **config['optimizer'])

    writer = SummaryWriter(log_dir=os.path.join('runs','fcn_'+datetime.now().strftime("%Y%m%d-%H%M%S")))
    _create_model_training_folder(writer, files_to_same=["./config_fcn.yaml","./downstream_fcn.py"])
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')

    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    global_step = 0
    max_epochs = config['trainer']['max_epochs']
    for epoch_counter in range(1, max_epochs+1):
        fcn.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch_counter}/{max_epochs}', unit='img') as pbar:
            cnt = 0
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                # image size [b, channels, H, W] float32; true_masks [b, H, W] in long
                # pred_masks [b, class, H, W] float32
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                with torch.cuda.amp.autocast():
                    pred_masks = fcn(images)
                    loss1 = criterion(pred_masks.squeeze(1), true_masks.float())
                    loss2 = focal_tversky_loss(torch.sigmoid(pred_masks.squeeze(1)), 
                        true_masks.float(), multiclass=False, alpha=0.3, beta=0.7, gamma=2.)
                    loss3 = criterionH(F.log_softmax(pred_masks, dim=1), true_masks[:,None,:,:])
                    lamdaF = loss2.item()/loss1.item()
                    lamdaH = loss2.item()/loss3.item()
                    loss = lamdaF*loss1 + loss2 + lamdaH*loss3
                    cnt += 1

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(fcn.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'batch loss': loss1.item()+loss2.item()})
                writer.add_scalar('total loss', loss.item(), global_step=global_step)
                writer.add_scalar('FCE loss', loss1.item(), global_step=global_step)
                writer.add_scalar('FT loss', loss2.item(), global_step=global_step)
                writer.add_scalar('HD loss', loss3.item(), global_step=global_step)

                #division_step = (n_train // (5 * config['trainer']['batch_size']))
                division_step = (n_train // config['trainer']['batch_size'])
                if division_step > 0:
                    if global_step % division_step == 0:
                        lossFCE, lossFT, lossHD = evaluate(fcn, val_loader, device, criterion)
                        logging.info('GlobalStep {0} Epoch {1} Step {2} Validation loss: {3:8.5f} {4:8.5f} {5:11.5f}'.format(global_step, epoch_counter, cnt, lossFCE, lossFT, lossHD))
 
        torch.save({
           'fcn_state_dict': fcn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(model_checkpoints_folder, 'model_epoch'+str(epoch_counter)+'.pth') )

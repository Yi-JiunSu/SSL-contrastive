"""
  consolidated and modified from two sources by Yi-Jiun Su 
     https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py
     https://github.com/lucidrains/pixel-level-contrastive-learning/blob/main/pixel_level_contrastive_learning/pixel_level_contrastive_learning.py

  The latest update in April 2024 by Y.-J. Su
"""
import torch
from torch import nn, einsum
import torch.nn.functional as F
from math import sqrt, floor
from copy import deepcopy
import random
from einops import rearrange
import os
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from datetime import datetime
import torchvision

def _create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))

def cutout_coordinateW(image, ratio_range = (0.7, 0.9)):
    _, _, h, orig_w = image.shape
    ratio_lo, ratio_hi = ratio_range
    random_ratio = ratio_lo + random.random() * (ratio_hi - ratio_lo)
    w = floor(random_ratio * orig_w)
    coor_x = floor((orig_w - w) *random.random())
    return ((0,h), (coor_x, coor_x+w)), random_ratio

def cutout_and_resize(image, coordinates, output_size = None, mode = 'nearest'):
    output_size = image.shape[2:] if output_size is None else output_size
    (y0, y1), (x0, x1) = coordinates
    cutout_image = image[:, :, y0:y1, x0:x1]
    return F.interpolate(cutout_image, size = output_size, mode = mode)

# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# loss fn defined as mean squared error in Grill et al. [2020]
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# pairwise_angle considered as part of the loss function 
def pairwise_angle(x, y):
    dotprod = torch.einsum('ij,ij->i', x, y)
    x = torch.linalg.norm(x,axis=1)
    y = torch.linalg.norm(y,axis=1)
    return torch.arccos(dotprod/x/y)

# Multi-Layer Perceptorn for instance-level projector
class MLP(nn.Module):
    def __init__(self, chan, chan_out = 256, inner_dim = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(chan, inner_dim),
            nn.BatchNorm1d(inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, chan_out)
        )
    def forward(self, x):
        return self.net(x)

# Pixel-to-Propagation Module (PPM) - the last step of online network of PixPro
class PPM(nn.Module):
    def __init__(
        self,
        *,
        chan,
        num_layers = 1,
        gamma = 2):
        super().__init__()
        self.gamma = gamma
        if num_layers == 0:
            self.transform_net = nn.Identity()
        elif num_layers == 1:
            self.transform_net = nn.Conv2d(chan, chan, 1)
        elif num_layers == 2:
            self.transform_net = nn.Sequential(
                nn.Conv2d(chan, chan, 1),
                nn.BatchNorm2d(chan),
                nn.ReLU(),
                nn.Conv2d(chan, chan, 1)
            )
        else:
            raise ValueError('num_layers must be one of 0, 1, or 2')
    def forward(self, x):
        xi = x[:, :, :, :, None, None]
        xj = x[:, :, None, None, :, :]
        similarity = F.relu(F.cosine_similarity(xi, xj, dim = 1)) ** self.gamma
        transform_out = self.transform_net(x)
        out = einsum('b x y h w, b c h w -> b c x y', similarity, transform_out)
        return out

# Multi-Layer Perceptron for pixel-level projector
class ConvMLP(nn.Module):
    def __init__(self, chan, chan_out = 256, inner_dim = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, inner_dim, 1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(),
            nn.Conv2d(inner_dim, chan_out, 1)
        )
    def forward(self, x):
        return self.net(x)

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets
# ResNet Layer_ID
#   0 -10'conv1' - Conv2d
#   1 -9 'bn1' - BatchNorm2d
#   2 -8 'relu' - ReLu - c0
#   3 -7 'maxpool' - MaxPool2d - c1
#   4 -6 'layer1' - c2
#   5 -5 'layer2' - c3
#   6 -4 'layer3' - c4
#   7 -3 'layer4' - pixel_layer
#   8 -2 'avgpool - instance_layer
#   9 -1 'fc'
class NetWrapperMultiLayers(nn.Module):
    def __init__(
        self, 
        *,
        net, 
        layer_IDs = [2,3,4,5,6,7,8]
    ):
        super().__init__()
        self.net = net
        self.layer_IDs = layer_IDs
        self.num_layers = len(layer_IDs)
        
        self.hook_registered = False 
        self.hidden_c0 = None
        self.hidden_c1 = None
        self.hidden_c2 = None
        self.hidden_c3 = None
        self.hidden_c4 = None
        self.hidden_pixel = None
        self.hidden_instance = None
    """
       layer_ids can be numbers or strings
    """
    def _find_layer(self, layer_id):
        if type(layer_id) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(layer_id, None)
        elif type(layer_id) == int:
            children = [*self.net.children()]
            return children[layer_id]
        return None

    # assigned label to the output for specific hidden layer
    def _hook_c0(self, _, __, output):
        setattr(self, 'hidden_c0', output)
    def _hook_c1(self, _, __, output):
        setattr(self, 'hidden_c1', output)
    def _hook_c2(self, _, __, output):
        setattr(self, 'hidden_c2', output)
    def _hook_c3(self, _, __, output):
        setattr(self, 'hidden_c3', output)
    def _hook_c4(self, _, __, output):
        setattr(self, 'hidden_c4', output)
    def _hook_pixel(self, _, __, output):
        setattr(self, 'hidden_pixel', output)
    def _hook_instance(self, _, __, output):
        setattr(self, 'hidden_instance', output)

    def _register_hook(self):
        for i in range(self.num_layers):
            match self.layer_IDs[i]:
                case item if item in [2, -8, 'relu']:
                    c0_layer = self._find_layer(self.layer_IDs[i])
                    assert c0_layer is not None, f'hidden layer ({self.layer_IDs[i]}) not found'
                    c0_layer.register_forward_hook(self._hook_c0)
                case item if item in [3, -7, 'maxpool']:
                    c1_layer = self._find_layer(self.layer_IDs[i])
                    assert c1_layer is not None, f'hidden layer ({self.layer_IDs[i]}) not found'
                    c1_layer.register_forward_hook(self._hook_c1)
                case item if item in [4, -6, 'layer1']:
                    c2_layer = self._find_layer(self.layer_IDs[i])
                    assert c2_layer is not None, f'hidden layer ({self.layer_IDs[i]}) not found'
                    c2_layer.register_forward_hook(self._hook_c2)
                case item if item in [5, -5, 'layer2']:
                    c3_layer = self._find_layer(self.layer_IDs[i])
                    assert c3_layer is not None, f'hidden layer ({self.layer_IDs[i]}) not found'
                    c3_layer.register_forward_hook(self._hook_c3)
                case item if item in [6, -4, 'layer3']:
                    c4_layer = self._find_layer(self.layer_IDs[i])
                    assert c4_layer is not None, f'hidden layer ({self.layer_IDs[i]}) not found'
                    c4_layer.register_forward_hook(self._hook_c4)
                case item if item in [7, -3, 'layer4']:
                    pixel_layer = self._find_layer(self.layer_IDs[i])
                    assert pixel_layer is not None, f'hidden layer ({self.layer_IDs[i]}) not found'
                    pixel_layer.register_forward_hook(self._hook_pixel)
                case item if item in [8, -2, 'avgpool']:
                    instance_layer = self._find_layer(self.layer_IDs[i])
                    assert instance_layer is not None, f'hidden layer ({self.layer_IDs[i]}) not found'
                    instance_layer.register_forward_hook(self._hook_instance)
                case item if item in [0, 1, 9, -10, -9, -1, 'conv1', 'bn1', 'fc']:
                    print(f'hidden layer ({self.layer_IDs[i]}) not specified for output')
                case _:
                    assert self._find_layer(self.layer_IDs[i]) is not None, f'hidden layer ({self.layer_IDs[i]}) not found'
        self.hook_registered = True

    def get_representation_multi(self, x):
        if not self.hook_registered:
            self._register_hook()
        _ = self.net(x)
        hidden_c0 = self.hidden_c0
        hidden_c1 = self.hidden_c1
        hidden_c2 = self.hidden_c2
        hidden_c3 = self.hidden_c3
        hidden_c4 = self.hidden_c4
        hidden_pixel = self.hidden_pixel
        hidden_instance = self.hidden_instance
        self.hidden_c0 = None
        self.hidden_c1 = None
        self.hidden_c2 = None
        self.hidden_c3 = None
        self.hidden_c4 = None
        self.hidden_pixel = None
        self.hidden_instance = None
        return hidden_instance, hidden_pixel, hidden_c4, hidden_c3, hidden_c2, hidden_c1, hidden_c0

    def get_representation(self, x):
        if not self.hook_registered:
            self._register_hook()
        _ = self.net(x)
        hidden_pixel = self.hidden_pixel
        hidden_instance = self.hidden_instance
        self.hidden_pixel = None
        self.hidden_instance = None
        return hidden_instance, hidden_pixel

    def forward(self, x, return_multi = False):
        if return_multi:
            return self.get_representation_multi(x)
        else:
            instance_rep, pixel_rep = self.get_representation(x)
            return instance_rep.flatten(1), pixel_rep

class BYOLTrainer():
    def __init__(
        self, 
        online_encoder,
        target_encoder, 
        online_predictor,
        optimizer, 
        scheduler, 
        device, 
        augment1, 
        augment2, 
        **params
    ):
        self.online_encoder = online_encoder
        self.target_encoder = target_encoder
        self.online_predictor = online_predictor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.augment1 = augment1
        self.augment2 = augment2
        self.alpha = 1.
        self.image_sizeH = params['image_sizeH']
        self.image_sizeW = params['image_sizeW']
        self.projection_size = params['projection_size']
        self.projection_hidden_size = params['projection_hidden_size']
        self.target_ema_updater = EMA(params['moving_average_decay'])
        self.batch_size = params['batch_size']
        self.max_epochs = params['max_epochs']
        self.num_workers = params['num_workers'] if params['num_workers'] != 'None' else os.cpu_count()
        self.pdict = nn.PairwiseDistance(p=2.0)
        self.writer = SummaryWriter(log_dir=os.path.join('runs','byol_'+datetime.now().strftime("%Y%m%d-%H%M%S")))
        _create_model_training_folder(self.writer, files_to_same=["./config_byol.yaml","./train_byol.py","./utils/pixcl_multi.py"])

    def train(self, train_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                num_workers=self.num_workers, drop_last=False, shuffle=True)
        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        for epoch_counter in range(self.max_epochs):
            lr_epoch = 0
            loss_epoch = 0
            for batch in train_loader:
                image0 = batch['image']
                image0 = image0.to(self.device)
                image1, image2 = self.augment1(image0), self.augment2(image0)
                image1 = image1.to(self.device)
                image2 = image2.to(self.device)

                if niter == 0:
                    grid = torchvision.utils.make_grid(image0[:32])
                    self.writer.add_image('image0', grid, global_step=niter)
                    grid = torchvision.utils.make_grid(image1[:32])
                    self.writer.add_image('image1', grid, global_step=niter)
                    grid = torchvision.utils.make_grid(image2[:32])
                    self.writer.add_image('image2', grid, global_step=niter)

                rep1, _ = self.online_encoder(image1)
                rep2, _ = self.online_encoder(image2)
                online_projector = MLP(rep1.shape[1],self.projection_size, self.projection_hidden_size).to(self.device)
                online_proj1, online_proj2 = online_projector(rep1), online_projector(rep2)
                online_pred1 = self.online_predictor(online_proj1)
                online_pred2 = self.online_predictor(online_proj2)

                with torch.no_grad():
                    target_rep1, _ = self.target_encoder(image1)
                    target_rep2, _ = self.target_encoder(image2)
                    online_projector = MLP(target_rep1.shape[1],self.projection_size, self.projection_hidden_size).to(self.device)
                    target_proj1, target_proj2 = online_projector(target_rep1), online_projector(target_rep2)
                    target_proj1.detach_()
                    target_proj2.detach_()

                d12 = self.pdict(online_pred1, target_proj2.detach())
                d21 = self.pdict(online_pred2, target_proj1.detach())
#                a_one = self.pairwise_angle(online_pred1, target_proj2.detach())
#                a_two = self.pairwise_angle(online_pred2, target_proj1.detach())
                a12 = pairwise_angle(online_pred1, target_proj2.detach())
                a21 = pairwise_angle(online_pred2, target_proj1.detach())

#                loss_one = loss_fn(online_pred1, target_proj2.detach())
#                loss_two = loss_fn(online_pred2, target_proj1.detach())

#                loss = (loss_one + loss_two).mean()
                d = (d12 + d21).mean()
                a = (a12 + a21).mean()
                loss =  d + a*self.alpha
                self.alpha = d.item()/a.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.writer.add_scalar('total loss', loss, global_step=niter)
                self.writer.add_scalar('distance loss', d, global_step=niter)
                self.writer.add_scalar('angle loss', a, global_step=niter)
                self.optimizer.step()
                loss_epoch += loss
                lr_epoch += self.optimizer.param_groups[0]['lr']
                update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
                niter += 1

            loss_epoch /= len(train_loader)
            lr_epoch /= len(train_loader)
            self.scheduler.step(loss_epoch)
            self.writer.add_scalar('epoch loss', loss_epoch, global_step=epoch_counter+1)
            self.writer.add_scalar('epoch learing rate', lr_epoch, global_step=epoch_counter+1)
            torch.save({
                'online_encoder_state_dict': self.online_encoder.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            },(os.path.join(model_checkpoints_folder, 'model_epoch'+str(epoch_counter+1)+'.pth')))
            print("End of epoch {}".format(epoch_counter+1))

class PixclLearner():
    def __init__(
        self,
        online_encoder, 
        target_encoder, 
        optimizer, 
        scheduler, 
        propagate_pixels, 
        online_predictor, 
        device, 
        augment1, 
        augment2,
        **params
    ):
        self.online_encoder = online_encoder
        self.target_encoder = target_encoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.propagate_pixels = propagate_pixels
        self.online_predictor = online_predictor
        self.device = device
        self.augment1 = augment1
        self.augment2 = augment2
        self.image_sizeH = params['image_sizeH']
        self.image_sizeW = params['image_sizeW']
        self.projection_size = params['projection_size']
        self.projection_hidden_size = params['projection_hidden_size']
        self.target_ema_updater = EMA(params['moving_average_decay'])
        self.distance_thres = params['distance_thres']
        self.similarity_temperature = params['similarity_temperature']
        self.alpha = params['alpha']
        self.alpha_instance=1.
        self.cutout_ratio_range = (0.6, 0.8)
        self.cutout_interpolate_mode = 'nearest' 
        self.coord_cutout_interpolate_mode = 'bilinear'
        self.batch_size = params['batch_size']
        self.max_epochs = params['max_epochs']
        self.pdict = nn.PairwiseDistance(p=2.0)
        self.num_workers = params['num_workers'] if params['num_workers'] != 'None' else os.cpu_count()
        self.writer = SummaryWriter(log_dir=os.path.join('runs','pixcl_'+datetime.now().strftime("%Y%m%d-%H%M%S")))
        _create_model_training_folder(self.writer, files_to_same=["./config_pixcl.yaml","./train_pixcl.py","./utils/pixcl_multi.py"])

    def train(self, train_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        for epoch_counter in range(self.max_epochs):
            loss_epoch = 0
            lr_epoch = 0
            for batch in train_loader:
                image0 = batch['image']
                image0 = image0.to(self.device)

                # data augmentation to generate two views from the original image
                cutout_coordinates_one, _ = cutout_coordinateW(image0, self.cutout_ratio_range)
                cutout_coordinates_two, _ = cutout_coordinateW(image0, self.cutout_ratio_range)
                #x [B, C, H, W]
                image1_cutout = cutout_and_resize(image0, cutout_coordinates_one, mode = self.cutout_interpolate_mode)
                image2_cutout = cutout_and_resize(image0, cutout_coordinates_two, mode = self.cutout_interpolate_mode)
                # image_xxx_cutout [B, C, H, W]
                image1_cutout, image2_cutout = self.augment1(image1_cutout), self.augment2(image2_cutout)
                image1_cutout = image1_cutout.to(self.device)
                image2_cutout = image2_cutout.to(self.device)

                if niter == 0:
                    grid = torchvision.utils.make_grid(image0[:32])
                    self.writer.add_image('image0', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(image1_cutout[:32])
                    self.writer.add_image('image1', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(image2_cutout[:32])
                    self.writer.add_image('image2', grid, global_step=niter)
                """
                    make_grid(tensor[:32]) is to arrange 32 images in a row
                    if batch size is less than 32 output all images in a mini-batch
                    if the batch size is more than 32 output the first 32 images of a mini-batch
                """
                # output two layers (c6 & c5) from the online_encoder, i.e. resnet
                instance1, pixel1 = self.online_encoder(image1_cutout)
                instance2, pixel2 = self.online_encoder(image2_cutout)

                # applied projectors for the two layers
                pixel_projector = ConvMLP(pixel1.shape[1],self.projection_size, self.projection_hidden_size).to(self.device)
                instance_projector = MLP(instance1.shape[1],self.projection_size, self.projection_hidden_size).to(self.device)
                proj_instance1, proj_pixel1 = instance_projector(instance1), pixel_projector(pixel1)
                proj_instance2, proj_pixel2 = instance_projector(instance2), pixel_projector(pixel2)

                image_h, image_w = image0.shape[2:]

                proj_image_shape = proj_pixel1.shape[2:]
                proj_image_h, proj_image_w = proj_image_shape

                coordinates = torch.meshgrid(
                    torch.arange(image_h, device = image0.device),
                    torch.arange(image_w, device = image0.device), 
                    indexing='ij')

                coordinates = torch.stack(coordinates).unsqueeze(0).float()
                coordinates /= sqrt(image_h ** 2 + image_w ** 2)
                coordinates[:, 0] *= proj_image_h
                coordinates[:, 1] *= proj_image_w

                proj_coors_one = cutout_and_resize(coordinates, cutout_coordinates_one, output_size = proj_image_shape, mode = self.coord_cutout_interpolate_mode)
                proj_coors_two = cutout_and_resize(coordinates, cutout_coordinates_two, output_size = proj_image_shape, mode = self.coord_cutout_interpolate_mode)
                # proj_coors_xxx [1, 2,image_h,image_w]

                proj_coors_one, proj_coors_two = map(lambda t: rearrange(t, 'b c h w -> (b h w) c'), (proj_coors_one, proj_coors_two))
                # proj_coorx_xxx [image_h*image_w, 2]
                pdist = nn.PairwiseDistance(p = 2)

                num_pixels = proj_coors_one.shape[0]

                proj_coors_one_expanded = proj_coors_one[:, None].expand(num_pixels, num_pixels, -1).reshape(num_pixels * num_pixels, 2)
                proj_coors_two_expanded = proj_coors_two[None, :].expand(num_pixels, num_pixels, -1).reshape(num_pixels * num_pixels, 2)
                #proj_coors_xxx_expanded [(image_h*image_w)*(image_h*image_w), 2]

                distance_matrix = pdist(proj_coors_one_expanded, proj_coors_two_expanded)
                # distance_matrix [(image_h*image_w)*(image_h*image_w)]
                distance_matrix = distance_matrix.reshape(num_pixels, num_pixels)
                # distance_matrix [image_h*image_w, image_h*image_w]

                positive_mask_one_two = distance_matrix < self.distance_thres
                positive_mask_two_one = positive_mask_one_two.t()

                # applied target_encoder to output two layers (c6 & c5)
                # obtain target projectors on the two layers
                with torch.no_grad():
                    target_instance1, target_pixel1 = self.target_encoder(image1_cutout)
                    target_instance2, target_pixel2 = self.target_encoder(image2_cutout)
                    pixel_projector = ConvMLP(target_pixel1.shape[1],self.projection_size, self.projection_hidden_size).to(self.device)
                    instance_projector = MLP(target_instance1.shape[1],self.projection_size, self.projection_hidden_size).to(self.device)
                    target_proj_instance1, target_proj_pixel1 = instance_projector(target_instance1), pixel_projector(target_pixel1)
                    target_proj_instance2, target_proj_pixel2 = instance_projector(target_instance2), pixel_projector(target_pixel2)

                # flatten all the pixel projections
                flatten = lambda t: rearrange(t, 'b c h w -> b c (h w)')
                target_proj_pixel1, target_proj_pixel2 = list(map(flatten, (target_proj_pixel1, target_proj_pixel2)))
                # target_proj_pixel_xxx [B, projection_size, 3x4]

                # get total number of positive pixel pairs
                positive_pixel_pairs = positive_mask_one_two.sum()

                # applied online_predictor on the instance projection
                # get instance level loss
                pred_instance1 = self.online_predictor(proj_instance1)
                pred_instance2 = self.online_predictor(proj_instance2)
                # pred_instance_xxx [B, projection_size]
#                loss_instance_one_two = loss_fn(pred_instance1, target_proj_instance2.detach())
#                loss_instance_two_one = loss_fn(pred_instance2, target_proj_instance1.detach())
                # loss_instance_xxx [B]
#                instance_loss = (loss_instance_one_two + loss_instance_two_one).mean()

                d12 = self.pdict(pred_instance1, target_proj_instance2.detach())
                d21 = self.pdict(pred_instance2, target_proj_instance1.detach())
                a12 = pairwise_angle(pred_instance1, target_proj_instance2.detach())
                a21 = pairwise_angle(pred_instance2, target_proj_instance1.detach())
                d = (d12 + d21).mean()
                a = (a12 + a21).mean()
                instance_loss =  d + a*self.alpha_instance
                self.alpha_instance = d.item()/a.item()

                # applied pixel propagator on the pixel projection 
                # calculate pix pro loss
                propagated_pixels1 = self.propagate_pixels(proj_pixel1)
                propagated_pixels2 = self.propagate_pixels(proj_pixel2)
                # propagated_pixels_xxx [B, projection_size, 3, 4]
                propagated_pixels1, propagated_pixels2 = list(map(flatten, (propagated_pixels1, propagated_pixels2)))
                # propagated_pixels_xxx [B, projection_size, 12]

                propagated_similarity_one_two = F.cosine_similarity(propagated_pixels1[..., :, None], target_proj_pixel2[..., None, :], dim = 1)
                propagated_similarity_two_one = F.cosine_similarity(propagated_pixels2[..., :, None], target_proj_pixel1[..., None, :], dim = 1)
                #propagated_similarity_xxx_xxx [B, 12, 12]

                loss_pixpro_one_two = propagated_similarity_one_two.masked_select(positive_mask_one_two[None, ...]).mean()
                loss_pixpro_two_one = propagated_similarity_two_one.masked_select(positive_mask_two_one[None, ...]).mean()
                pixpro_loss = 2 - loss_pixpro_one_two - loss_pixpro_two_one

                # total loss
                loss = pixpro_loss*self.alpha + instance_loss
                self.alpha = instance_loss.item()/pixpro_loss.item()
        
                self.optimizer.zero_grad()
                loss.backward()
                self.writer.add_scalar('loss_total', loss, global_step=niter)
                self.writer.add_scalar('loss_pixpro', pixpro_loss, global_step=niter)
                self.writer.add_scalar('loss_instance', instance_loss, global_step=niter)
                self.writer.add_scalar('loss_dist', d, global_step=niter)
                self.writer.add_scalar('loss_angle', a, global_step=niter)
                self.optimizer.step()
                update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
                loss_epoch += loss
                lr_epoch += self.optimizer.param_groups[0]['lr']
                niter += 1

            loss_epoch /= len(train_loader)
            lr_epoch /= len(train_loader)
            self.scheduler.step(loss_epoch)
            self.writer.add_scalar('epoch loss', loss_epoch, global_step=epoch_counter+1)
            self.writer.add_scalar('epoch learning rate', lr_epoch, global_step=epoch_counter+1)
            torch.save({
                'online_encoder_state_dict': self.online_encoder.state_dict(),
#                'target_encoder_state_dict': self.target_encoder.state_dict(),
#                'propagate_pixels_state_dict': self.propagate_pixels.state_dict(),
#                'online_predictor_state_dict': self.online_predictor.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            },(os.path.join(model_checkpoints_folder, 'model_epoch'+str(epoch_counter+1)+'.pth')))
            print("End of epoch {}".format(epoch_counter+1))

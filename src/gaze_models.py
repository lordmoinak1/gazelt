import argparse
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm

from einops import rearrange, reduce

from global_block import *
from focal_block import *
from datasets import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class GlobalBlock(torch.nn.Module):
    def __init__(self):
        super(GlobalBlock, self).__init__()

        self.global_encoder = gc_vit_tiny(pretrained=False)

        self.global_decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=32**2 * 3,
                kernel_size=1,
            ),
            nn.PixelShuffle(32),)
        
        self.global_decoder_0 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=32**2 * 3,
                kernel_size=1,
            ),
            nn.PixelShuffle(32),
            nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            )
        self.global_decoder_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=32**2 * 3,
                kernel_size=1,
            ),
            nn.PixelShuffle(32),
            nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            )
        self.global_decoder_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=32**2 * 3,
                kernel_size=1,
            ),
            nn.PixelShuffle(32),)
        self.global_decoder_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=32**2 * 3,
                kernel_size=1,
            ),
            nn.PixelShuffle(32),)

    def forward(self, x):
        layer_outputs, x_feat = self.global_encoder(x)

        layer_0 = rearrange(layer_outputs[0], 'b h w c -> b c h w')
        layer_1 = rearrange(layer_outputs[1], 'b h w c -> b c h w')
        layer_2 = rearrange(layer_outputs[2], 'b h w c -> b c h w')
        layer_3 = rearrange(layer_outputs[3], 'b h w c -> b c h w')

        layer_0 = self.global_decoder_0(layer_0)
        layer_1 = layer_0 + self.global_decoder_1(layer_1)
        layer_2 = layer_1 + self.global_decoder_2(layer_2)
        layer_3 = layer_2 + self.global_decoder_3(layer_3)

        return layer_0, layer_1, layer_2, layer_3, x_feat
    
class FocalBlock(torch.nn.Module):
    def __init__(self):
        super(FocalBlock, self).__init__()

        self.focal_encoder = focalnet_tiny_srf(pretrained=False)

        self.focal_decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=768,
                out_channels=32**2 * 3,
                kernel_size=1,
            ),
            nn.PixelShuffle(32),)
        
        self.focal_decoder_0 = nn.Sequential(
            nn.Conv2d(
                in_channels=192,
                out_channels=32**2 * 3,
                kernel_size=1,
            ),
            nn.PixelShuffle(32),
            nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),)
        self.focal_decoder_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=384,
                out_channels=32**2 * 3,
                kernel_size=1,
            ),
            nn.PixelShuffle(32),
            nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),)
        self.focal_decoder_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=768,
                out_channels=32**2 * 3,
                kernel_size=1,
            ),
            nn.PixelShuffle(32),)
        self.focal_decoder_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=768,
                out_channels=32**2 * 3,
                kernel_size=1,
            ),
            nn.PixelShuffle(32),)

    def forward(self, x):
        layer_outputs, x_feat = self.focal_encoder(x)

        layer_0 = reduce(layer_outputs[0], 'b (h w) c -> b h w c', 'mean', h=28, w=28)
        layer_1 = reduce(layer_outputs[1], 'b (h w) c -> b h w c', 'mean', h=14, w=14)
        layer_2 = reduce(layer_outputs[2], 'b (h w) c -> b h w c', 'mean', h=7, w=7)
        layer_3 = reduce(layer_outputs[3], 'b (h w) c -> b h w c', 'mean', h=7, w=7)

        layer_0 = rearrange(layer_0, 'b h w c -> b c h w')
        layer_1 = rearrange(layer_1, 'b h w c -> b c h w')
        layer_2 = rearrange(layer_2, 'b h w c -> b c h w')
        layer_3 = rearrange(layer_3, 'b h w c -> b c h w')

        layer_0 = self.focal_decoder_0(layer_0)
        layer_1 = layer_0 + self.focal_decoder_1(layer_1)
        layer_2 = layer_1 + self.focal_decoder_2(layer_2)
        layer_3 = layer_2 + self.focal_decoder_3(layer_3)
        
        return layer_0, layer_1, layer_2, layer_3, x_feat

class GlobalTeacher(torch.nn.Module):
    def __init__(self, data_path=None):
        super(GlobalTeacher, self).__init__()
        self.global_encoder_x = GlobalBlock()
        self.global_encoder_x.load_state_dict(torch.load(data_path))
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.global_linear = nn.Linear(512, 64)

    def forward(self, x):
        with torch.no_grad():
            _, _, _, _, x_global = self.global_encoder_x(x)
        x_global = rearrange(x_global, 'b h w c -> b (h w) c')
        x_global = self.global_avgpool(x_global.transpose(1, 2))  # B C 1
        x_global = self.global_linear(x_global.squeeze()) 
        return x_global
    
class FocalTeacher(torch.nn.Module):
    def __init__(self, data_path=None):
        super(FocalTeacher, self).__init__()
        self.focal_encoder_x = FocalBlock()
        self.focal_encoder_x.load_state_dict(torch.load(data_path))
        self.focal_avgpool = nn.AdaptiveAvgPool1d(1)
        self.focal_linear = nn.Linear(768, 64)

    def forward(self, x):
        with torch.no_grad():
            _, _, _, _, x_focal = self.focal_encoder_x(x)
        x_focal = rearrange(x_focal, 'b h w c -> b (h w) c')
        x_focal = self.focal_avgpool(x_focal.transpose(1, 2))  # B C 1
        x_focal = self.focal_linear(x_focal.squeeze()) 
        return x_focal
    
def generate_features_global_nih(model_path=None, data_path=None, labels_path=None, save_path=None, mode=None):
    model = GlobalTeacher(model_path)
    model = model.to(device)

    train_dataset = NIH_CXR_Dataset_FE(data_dir=data_path, label_dir=labels_path, split='train')
    val_dataset = NIH_CXR_Dataset_FE(data_dir=data_path, label_dir=labels_path, split='balanced-val')
    test_dataset = NIH_CXR_Dataset_FE(data_dir=data_path, label_dir=labels_path, split='test')
    bal_test_dataset = NIH_CXR_Dataset_FE(data_dir=data_path, label_dir=labels_path, split='balanced-test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True) #, worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True) #), worker_init_fn=val_worker_init_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True) #), worker_init_fn=val_worker_init_fn)
    bal_test_loader = torch.utils.data.DataLoader(bal_test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True) #), worker_init_fn=val_worker_init_fn)

    if mode == 'bal':
        for images, _, name in tqdm(bal_test_loader):
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
            np.save(os.path.join(save_path, 'global_{}.npy'.format(name[0])), outputs.detach().cpu().numpy())
    else:
        for images, _, name in tqdm(test_loader):
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
            np.save(os.path.join(save_path, 'global_{}.npy'.format(name[0])), outputs.detach().cpu().numpy())

        for images, _, name in tqdm(train_loader):
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
            np.save(os.path.join(save_path, 'global_{}.npy'.format(name[0])), outputs.detach().cpu().numpy())

        for images, _, name in tqdm(val_loader):
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
            np.save(os.path.join(save_path, 'global_{}.npy'.format(name[0])), outputs.detach().cpu().numpy())

def generate_features_focal_nih(model_path=None, data_path=None, labels_path=None, save_path=None, mode=None):
    model = FocalTeacher(model_path)
    model = model.to(device)

    train_dataset = NIH_CXR_Dataset_FE(data_dir=data_path, label_dir=labels_path, split='train')
    val_dataset = NIH_CXR_Dataset_FE(data_dir=data_path, label_dir=labels_path, split='balanced-val')
    test_dataset = NIH_CXR_Dataset_FE(data_dir=data_path, label_dir=labels_path, split='test')
    bal_test_dataset = NIH_CXR_Dataset_FE(data_dir=data_path, label_dir=labels_path, split='balanced-test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True) #, worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True) #), worker_init_fn=val_worker_init_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True) #), worker_init_fn=val_worker_init_fn)
    bal_test_loader = torch.utils.data.DataLoader(bal_test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True) #), worker_init_fn=val_worker_init_fn)

    if mode == 'bal':
        for images, _, name in tqdm(bal_test_loader):
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
            np.save(os.path.join(save_path, 'focal_{}.npy'.format(name[0])), outputs.detach().cpu().numpy())
    else:
        for images, _, name in tqdm(test_loader):
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
            np.save(os.path.join(save_path, 'focal_{}.npy'.format(name[0])), outputs.detach().cpu().numpy())

        for images, _, name in tqdm(train_loader):
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
            np.save(os.path.join(save_path, 'focal_{}.npy'.format(name[0])), outputs.detach().cpu().numpy())

        for images, _, name in tqdm(val_loader):
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
            np.save(os.path.join(save_path, 'focal_{}.npy'.format(name[0])), outputs.detach().cpu().numpy())

def generate_features_global_mimic(model_path=None, data_path=None, labels_path=None, save_path=None, mode=None):
    model = GlobalTeacher(model_path)
    model = model.to(device)
    model.eval()

    train_dataset = MIMIC_CXR_Dataset_FE(data_dir=data_path, label_dir=labels_path, split='train')
    val_dataset = MIMIC_CXR_Dataset_FE(data_dir=data_path, label_dir=labels_path, split='balanced-val')
    test_dataset = MIMIC_CXR_Dataset_FE(data_dir=data_path, label_dir=labels_path, split='test')
    bal_test_dataset = MIMIC_CXR_Dataset_FE(data_dir=data_path, label_dir=labels_path, split='balanced-test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True) #, worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True) #), worker_init_fn=val_worker_init_fn)
    bal_test_loader = torch.utils.data.DataLoader(bal_test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True) #), worker_init_fn=val_worker_init_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True) #), worker_init_fn=val_worker_init_fn)

    if mode == 'bal':
        for images, _, name in tqdm(bal_test_loader):
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
            np.save(os.path.join(save_path, 'global_{}.npy'.format(name[0])), outputs.detach().cpu().numpy())
    else:
        for images, _, name in tqdm(test_loader):
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
            np.save(os.path.join(save_path, 'global_{}.npy'.format(name[0])), outputs.detach().cpu().numpy())

        for images, _, name in tqdm(train_loader):
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
            np.save(os.path.join(save_path, 'global_{}.npy'.format(name[0])), outputs.detach().cpu().numpy())

        for images, _, name in tqdm(val_loader):
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
            np.save(os.path.join(save_path, 'global_{}.npy'.format(name[0])), outputs.detach().cpu().numpy())

def generate_features_focal_mimic(model_path=None, data_path=None, labels_path=None, save_path=None, mode=None):
    model = FocalTeacher(model_path)
    model = model.to(device)
    model.eval()

    train_dataset = MIMIC_CXR_Dataset_FE(data_dir=data_path, label_dir=labels_path, split='train')
    val_dataset = MIMIC_CXR_Dataset_FE(data_dir=data_path, label_dir=labels_path, split='balanced-val')
    test_dataset = MIMIC_CXR_Dataset_FE(data_dir=data_path, label_dir=labels_path, split='test')
    bal_test_dataset = MIMIC_CXR_Dataset_FE(data_dir=data_path, label_dir=labels_path, split='balanced-test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True) #, worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True) #), worker_init_fn=val_worker_init_fn)
    bal_test_loader = torch.utils.data.DataLoader(bal_test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True) #), worker_init_fn=val_worker_init_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True) #), worker_init_fn=val_worker_init_fn)


    if mode == 'bal':
        for images, _, name in tqdm(bal_test_loader):
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
            np.save(os.path.join(save_path, 'focal_{}.npy'.format(name[0])), outputs.detach().cpu().numpy())
    else:
        for images, _, name in tqdm(test_loader):
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
            np.save(os.path.join(save_path, 'focal_{}.npy'.format(name[0])), outputs.detach().cpu().numpy())

        for images, _, name in tqdm(train_loader):
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
            np.save(os.path.join(save_path, 'focal_{}.npy'.format(name[0])), outputs.detach().cpu().numpy())

        for images, _, name in tqdm(val_loader):
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
            np.save(os.path.join(save_path, 'focal_{}.npy'.format(name[0])), outputs.detach().cpu().numpy())

def main():
    parser = argparse.ArgumentParser(description="Feature Generator for NIH and MIMIC datasets")
    
    parser.add_argument('--global', dest='global_features', action='store_true',
                        help='Generate global features')
    parser.add_argument('--focal', dest='focal_features', action='store_true',
                        help='Generate focal features')
    parser.add_argument('--nih', dest='use_nih', action='store_true',
                        help='Use NIH dataset')
    parser.add_argument('--mimic', dest='use_mimic', action='store_true',
                        help='Use MIMIC dataset')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model weights')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the image dataset')
    parser.add_argument('--labels_path', type=str, required=True,
                        help='Path to the label files')
    parser.add_argument('--save_features_path', type=str, required=True,
                        help='Path to save extracted features')
    parser.add_argument('--mode', type=str, required=False,
                    help='mode')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.save_features_path, exist_ok=True)

    if args.global_features and args.use_nih:
        generate_features_global_nih(args.model_path, args.data_path, args.labels_path, args.save_features_path, args.mode)
    elif args.focal_features and args.use_nih:
        generate_features_focal_nih(args.model_path, args.data_path, args.labels_path, args.save_features_path, args.mode)
    elif args.global_features and args.use_mimic:
        generate_features_global_mimic(args.model_path, args.data_path, args.labels_path, args.save_features_path, args.mode)
    elif args.focal_features and args.use_mimic:
        generate_features_focal_mimic(args.model_path, args.data_path, args.labels_path, args.save_features_path, args.mode)
    else:
        print("❌ Please specify one of: --global or --focal and one of: --nih or --mimic")

if __name__ == "__main__":
    main()

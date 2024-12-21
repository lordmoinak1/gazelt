import os

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

class NIH_CXR_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split):
        self.data_dir = data_dir
        self.split = split

        self.CLASSES = [
            'No Finding', 'Infiltration', 'Atelectasis', 'Effusion', 'Nodule',
            'Mass', 'Pneumothorax', 'Consolidation', 'Pleural_Thickening',
            'Cardiomegaly', 'Fibrosis', 'Edema', 'Tortuous Aorta', 'Emphysema',
            'Pneumonia', 'Calcification of the Aorta', 'Pneumoperitoneum', 'Hernia',
            'Subcutaneous Emphysema', 'Pneumomediastinum'
        ]

        self.label_df = pd.read_csv(os.path.join(label_dir, f'nih-cxr-lt_single-label_{split}.csv'))

        self.img_paths = self.label_df['id'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        self.labels = self.label_df[self.CLASSES].idxmax(axis=1).apply(lambda x: self.CLASSES.index(x)).values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

        if self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.img_paths[idx])
        x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA)

        x = self.transform(x)

        y = np.array(self.labels[idx])

        name = self.img_paths[idx].split('/')[-1]
        name = name.split('.png')[0]

        return x.float(), torch.from_numpy(y).long()#, name

class MIMIC_CXR_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split):
        self.split = split

        self.CLASSES = [
            'No Finding', 'Lung Opacity', 'Cardiomegaly', 'Atelectasis',
            'Pleural Effusion', 'Support Devices', 'Edema', 'Pneumonia',
            'Pneumothorax', 'Lung Lesion', 'Fracture', 'Enlarged Cardiomediastinum',
            'Consolidation', 'Pleural Other', 'Calcification of the Aorta',
            'Tortuous Aorta', 'Pneumoperitoneum', 'Subcutaneous Emphysema',
            'Pneumomediastinum'
        ]

        self.label_df = pd.read_csv(os.path.join(label_dir, f'mimic-cxr-lt_single-label_{split}.csv'))

        self.img_paths = self.label_df['path'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        df = pd.DataFrame(self.img_paths)
        df.to_csv('/home/moibhattacha/LongTailCXR/mimic_image_list_{}.csv'.format(split))
        self.labels = self.label_df[self.CLASSES].idxmax(axis=1).apply(lambda x: self.CLASSES.index(x)).values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

        if self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.img_paths[idx])
        x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA)
        x = self.transform(x)

        y = np.array(self.labels[idx])

        # print(self.img_paths[idx])

        name = self.img_paths[idx].split('/')[-1]
        name = name.split('.jpg')[0]

        return x.float(), torch.from_numpy(y).long()#, name

## CREDIT TO https://github.com/agaldran/balanced_mixup ##

# pytorch-wrapping-multi-dataloaders/blob/master/wrapping_multi_dataloaders.py
class ComboIter(object):
    """An iterator."""
    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [loader_iter.next() for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)

class ComboLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.
    Args:
    loaders: a list or tuple of pytorch DataLoader objects
    """
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return ComboIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches
    
class NIH_CXR_Teacher_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split):
        self.data_dir = data_dir
        self.split = split

        self.CLASSES = [
            'No Finding', 'Infiltration', 'Atelectasis', 'Effusion', 'Nodule',
            'Mass', 'Pneumothorax', 'Consolidation', 'Pleural_Thickening',
            'Cardiomegaly', 'Fibrosis', 'Edema', 'Tortuous Aorta', 'Emphysema',
            'Pneumonia', 'Calcification of the Aorta', 'Pneumoperitoneum', 'Hernia',
            'Subcutaneous Emphysema', 'Pneumomediastinum'
        ]

        self.label_df = pd.read_csv(os.path.join(label_dir, f'nih-cxr-lt_single-label_{split}.csv'))

        self.img_paths = self.label_df['id'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        self.labels = self.label_df[self.CLASSES].idxmax(axis=1).apply(lambda x: self.CLASSES.index(x)).values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

        if self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.img_paths[idx])
        x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA)

        x = self.transform(x)

        y = np.array(self.labels[idx])

        name = self.img_paths[idx].split('/')[-1]
        name = name.split('.png')[0]

        focal_features = np.load('/path/to/model_outputs/gaze_lt/focal_{}.npy'.format(name))#.detach().cpu()
        global_features = np.load('/path/to/model_outputs/gaze_lt/global_{}.npy'.format(name))#.detach().cpu()

        # focal_features = np.load('/path/to/gazelt/features/nih/focal_{}.npy'.format(name))#.detach().cpu()
        # global_features = np.load('/path/to/gazelt/features/nih/global_{}.npy'.format(name))#.detach().cpu()
        
        return x.float(), focal_features, global_features, torch.from_numpy(y).long()
    
class MIMIC_CXR_Teacher_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split):
        self.split = split

        self.CLASSES = [
            'No Finding', 'Lung Opacity', 'Cardiomegaly', 'Atelectasis',
            'Pleural Effusion', 'Support Devices', 'Edema', 'Pneumonia',
            'Pneumothorax', 'Lung Lesion', 'Fracture', 'Enlarged Cardiomediastinum',
            'Consolidation', 'Pleural Other', 'Calcification of the Aorta',
            'Tortuous Aorta', 'Pneumoperitoneum', 'Subcutaneous Emphysema',
            'Pneumomediastinum'
        ]

        self.label_df = pd.read_csv(os.path.join(label_dir, f'mimic-cxr-lt_single-label_{split}.csv'))

        self.img_paths = self.label_df['path'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        df = pd.DataFrame(self.img_paths)
        df.to_csv('/home/moibhattacha/LongTailCXR/mimic_image_list_{}.csv'.format(split))
        self.labels = self.label_df[self.CLASSES].idxmax(axis=1).apply(lambda x: self.CLASSES.index(x)).values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

        if self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.img_paths[idx])
        x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA)
        x = self.transform(x)

        y = np.array(self.labels[idx])

        name = self.img_paths[idx].split('/')[-1]
        name = name.split('.jpg')[0]

        # focal_features = np.load('/path/to/model_outputs/gaze_lt_mimic/focal_{}.npy'.format(name))#.detach().cpu()
        # global_features = np.load('/path/to/model_outputs/gaze_lt_mimic/global_{}.npy'.format(name))#.detach().cpu()

        focal_features = np.load('/path/to/gazelt/features/mimic/focal_{}.npy'.format(name))#.detach().cpu()
        global_features = np.load('/path/to/gazelt/features/mimic/global_{}.npy'.format(name))#.detach().cpu()

        return x.float(), focal_features, global_features, torch.from_numpy(y).long()

class NIH_CXR_RadioTransformer_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split):
        self.data_dir = data_dir
        self.split = split

        self.CLASSES = [
            'No Finding', 'Infiltration', 'Atelectasis', 'Effusion', 'Nodule',
            'Mass', 'Pneumothorax', 'Consolidation', 'Pleural_Thickening',
            'Cardiomegaly', 'Fibrosis', 'Edema', 'Tortuous Aorta', 'Emphysema',
            'Pneumonia', 'Calcification of the Aorta', 'Pneumoperitoneum', 'Hernia',
            'Subcutaneous Emphysema', 'Pneumomediastinum'
        ]

        self.label_df = pd.read_csv(os.path.join(label_dir, f'nih-cxr-lt_single-label_{split}.csv'))

        self.img_paths = self.label_df['id'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        self.labels = self.label_df[self.CLASSES].idxmax(axis=1).apply(lambda x: self.CLASSES.index(x)).values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

        if self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.img_paths[idx])
        x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA)

        x = self.transform(x)

        y = np.array(self.labels[idx])

        name = self.img_paths[idx].split('/')[-1]
        name = name.split('.png')[0]

        radiotransformer_features = np.load('/path/to/model_outputs/radiotransformer/radiotransformer_{}.npy'.format(name))#.detach().cpu()
        radiotransformer_features = np.squeeze(radiotransformer_features, axis=0)
        
        return x.float(), radiotransformer_features, torch.from_numpy(y).long()
    
class NIH_CXR_GazeRadar_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split):
        self.data_dir = data_dir
        self.split = split

        self.CLASSES = [
            'No Finding', 'Infiltration', 'Atelectasis', 'Effusion', 'Nodule',
            'Mass', 'Pneumothorax', 'Consolidation', 'Pleural_Thickening',
            'Cardiomegaly', 'Fibrosis', 'Edema', 'Tortuous Aorta', 'Emphysema',
            'Pneumonia', 'Calcification of the Aorta', 'Pneumoperitoneum', 'Hernia',
            'Subcutaneous Emphysema', 'Pneumomediastinum'
        ]

        self.label_df = pd.read_csv(os.path.join(label_dir, f'nih-cxr-lt_single-label_{split}.csv'))

        self.img_paths = self.label_df['id'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        self.labels = self.label_df[self.CLASSES].idxmax(axis=1).apply(lambda x: self.CLASSES.index(x)).values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

        if self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.img_paths[idx])
        x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA)

        x = self.transform(x)

        y = np.array(self.labels[idx])

        name = self.img_paths[idx].split('/')[-1]
        name = name.split('.png')[0]

        gazeradar_features = np.load('/path/to/model_outputs/gazeradar/gazeradar_{}.npy'.format(name))#.detach().cpu()
        gazeradar_features = np.squeeze(gazeradar_features, axis=0)

        return x.float(), gazeradar_features, torch.from_numpy(y).long()
    
class MIMIC_CXR_RadioTransformer_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split):
        self.split = split

        self.CLASSES = [
            'No Finding', 'Lung Opacity', 'Cardiomegaly', 'Atelectasis',
            'Pleural Effusion', 'Support Devices', 'Edema', 'Pneumonia',
            'Pneumothorax', 'Lung Lesion', 'Fracture', 'Enlarged Cardiomediastinum',
            'Consolidation', 'Pleural Other', 'Calcification of the Aorta',
            'Tortuous Aorta', 'Pneumoperitoneum', 'Subcutaneous Emphysema',
            'Pneumomediastinum'
        ]

        self.label_df = pd.read_csv(os.path.join(label_dir, f'mimic-cxr-lt_single-label_{split}.csv'))

        self.img_paths = self.label_df['path'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        df = pd.DataFrame(self.img_paths)
        df.to_csv('/home/moibhattacha/LongTailCXR/mimic_image_list_{}.csv'.format(split))
        self.labels = self.label_df[self.CLASSES].idxmax(axis=1).apply(lambda x: self.CLASSES.index(x)).values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

        if self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.img_paths[idx])
        x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA)
        x = self.transform(x)

        y = np.array(self.labels[idx])

        # print(self.img_paths[idx])

        name = self.img_paths[idx].split('/')[-1]
        name = name.split('.jpg')[0]

        radiotransformer_features = np.load('/path/to/model_outputs/radiotransformer_mimic/radiotransformer_{}.npy'.format(name))#.detach().cpu()
        radiotransformer_features = np.squeeze(radiotransformer_features, axis=0)

        return x.float(), radiotransformer_features, torch.from_numpy(y).long()
    
class MIMIC_CXR_GazeRadar_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split):
        self.split = split

        self.CLASSES = [
            'No Finding', 'Lung Opacity', 'Cardiomegaly', 'Atelectasis',
            'Pleural Effusion', 'Support Devices', 'Edema', 'Pneumonia',
            'Pneumothorax', 'Lung Lesion', 'Fracture', 'Enlarged Cardiomediastinum',
            'Consolidation', 'Pleural Other', 'Calcification of the Aorta',
            'Tortuous Aorta', 'Pneumoperitoneum', 'Subcutaneous Emphysema',
            'Pneumomediastinum'
        ]

        self.label_df = pd.read_csv(os.path.join(label_dir, f'mimic-cxr-lt_single-label_{split}.csv'))

        self.img_paths = self.label_df['path'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        df = pd.DataFrame(self.img_paths)
        df.to_csv('/home/moibhattacha/LongTailCXR/mimic_image_list_{}.csv'.format(split))
        self.labels = self.label_df[self.CLASSES].idxmax(axis=1).apply(lambda x: self.CLASSES.index(x)).values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

        if self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.img_paths[idx])
        x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA)
        x = self.transform(x)

        y = np.array(self.labels[idx])

        # print(self.img_paths[idx])

        name = self.img_paths[idx].split('/')[-1]
        name = name.split('.jpg')[0]

        gazeradar_features = np.load('/path/to/model_outputs/gazeradar_mimic/gazeradar_{}.npy'.format(name))#.detach().cpu()
        gazeradar_features = np.squeeze(gazeradar_features, axis=0)
        
        return x.float(), gazeradar_features, torch.from_numpy(y).long()
    
class MIMIC_CXR_Dataset_GazeControlNet(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split):
        self.split = split

        self.CLASSES = [
            'No Finding', 'Lung Opacity', 'Cardiomegaly', 'Atelectasis',
            'Pleural Effusion', 'Support Devices', 'Edema', 'Pneumonia',
            'Pneumothorax', 'Lung Lesion', 'Fracture', 'Enlarged Cardiomediastinum',
            'Consolidation', 'Pleural Other', 'Calcification of the Aorta',
            'Tortuous Aorta', 'Pneumoperitoneum', 'Subcutaneous Emphysema',
            'Pneumomediastinum'
        ]

        if self.split == 'train':
            # self.label_df = pd.read_csv(os.path.join(label_dir, f'mimic-cxr-lt_single-label_{split}_controlnet.csv'))
            # self.label_df = pd.read_csv(os.path.join(label_dir, f'mimic-cxr-lt_single-label_{split}_multicontrolnet.csv'))
            self.label_df = pd.read_csv(os.path.join(label_dir, f'mimic-cxr-lt_single-label_{split}_gazecontrolnet.csv'))
        else:
            self.label_df = pd.read_csv(os.path.join(label_dir, f'mimic-cxr-lt_single-label_{split}.csv'))

        self.img_paths = self.label_df['path'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        df = pd.DataFrame(self.img_paths)
        # df.to_csv('/home/moibhattacha/LongTailCXR/mimic_image_list_{}.csv'.format(split))
        self.labels = self.label_df[self.CLASSES].idxmax(axis=1).apply(lambda x: self.CLASSES.index(x)).values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

        if self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.img_paths[idx])
        x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA)
        x = self.transform(x)

        y = np.array(self.labels[idx])

        name = self.img_paths[idx].split('/')[-1]
        name = name.split('.jpg')[0]

        return x.float(), torch.from_numpy(y).long()#, name
    
if __name__ == '__main__':
    flag = 0

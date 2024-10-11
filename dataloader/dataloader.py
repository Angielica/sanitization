import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import os
from PIL import Image

import natsort
class CustomDataset(Dataset):
    def __init__(self, data, transform, data_folder, type_set_folder, attack_type=None, params=None):
        self.data = data
        self.transform = transform
        self.data_folder = data_folder
        self.type_set_folder = type_set_folder
        self.attack_type = attack_type
        self.params = params

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clean_img_loc = os.path.join(self.params['DATA_FOLDER_CLEAN'], 'clean', self.type_set_folder, self.data[idx])
        clean_image = Image.open(clean_img_loc).convert("RGB")
        tensor_clean_image = self.transform(clean_image)

        mal_img_loc = os.path.join(self.data_folder, self.attack_type, self.type_set_folder, self.data[idx])
        malware_image = Image.open(mal_img_loc).convert("RGB")
        tensor_mal_image = self.transform(malware_image)

        return self.data[idx], tensor_mal_image, tensor_clean_image

def get_data(params):
    transform = transforms.Compose([transforms.ToTensor()])
    batch_size = params['batch_size']

    data_folder = params['DATA_FOLDER']
    data_folder_clean = params['DATA_FOLDER_CLEAN']
    train_data = natsort.natsorted(os.listdir(os.path.join(data_folder_clean, 'clean/train')))
    val_data = natsort.natsorted(os.listdir(os.path.join(data_folder_clean, 'clean/val')))
    test_data = natsort.natsorted(os.listdir(os.path.join(data_folder_clean, 'clean/test')))

    train_set_classic_rows = CustomDataset(data=train_data, transform=transform, data_folder=data_folder,
                                           type_set_folder='train', attack_type='lsb_classic/interleaving_rows',
                                           params=params)
    train_set_classic_squares = CustomDataset(data=train_data, transform=transform, data_folder=data_folder,
                                              type_set_folder='train', attack_type='lsb_classic/interleaving_squares',
                                           params=params)

    train_set_classic_sequential = CustomDataset(data=train_data, transform=transform, data_folder=data_folder,
                                                 type_set_folder='train', attack_type='lsb_classic/sequential',
                                           params=params)

    train_set_oceanlotus_rows = CustomDataset(data=train_data, transform=transform, data_folder=data_folder,
                                              type_set_folder='train',attack_type='lsb_oceanlotus/interleaving_rows',
                                           params=params)

    train_set_oceanlotus_squares = CustomDataset(data=train_data, transform=transform, data_folder=data_folder,
                                                 type_set_folder='train', attack_type='lsb_oceanlotus/interleaving_squares',
                                           params=params)

    train_set_oceanlotus_sequential = CustomDataset(data=train_data, transform=transform, data_folder=data_folder,
                                                    type_set_folder='train', attack_type='lsb_oceanlotus/sequential',
                                           params=params)

    val_set_classic_rows = CustomDataset(data=val_data, transform=transform, data_folder=data_folder,
                                         type_set_folder='val', attack_type='lsb_classic/interleaving_rows',
                                           params=params)

    val_set_classic_squares = CustomDataset(data=val_data, transform=transform, data_folder=data_folder,
                                            type_set_folder='val', attack_type='lsb_classic/interleaving_squares',
                                           params=params)

    val_set_classic_sequential = CustomDataset(data=val_data, transform=transform, data_folder=data_folder,
                                               type_set_folder='val', attack_type='lsb_classic/sequential',
                                           params=params)

    val_set_oceanlotus_rows = CustomDataset(data=val_data, transform=transform, data_folder=data_folder,
                                            type_set_folder='val', attack_type='lsb_oceanlotus/interleaving_rows',
                                           params=params)

    val_set_oceanlotus_squares = CustomDataset(data=val_data, transform=transform, data_folder=data_folder,
                                               type_set_folder='val', attack_type='lsb_oceanlotus/interleaving_squares',
                                           params=params)

    val_set_oceanlotus_sequential = CustomDataset(data=val_data, transform=transform, data_folder=data_folder,
                                                  type_set_folder='val', attack_type='lsb_oceanlotus/sequential',
                                           params=params)

    test_set_classic_rows = CustomDataset(data=test_data, transform=transform, data_folder=data_folder,
                                          type_set_folder='test', attack_type='lsb_classic/interleaving_rows',
                                           params=params)

    test_set_classic_squares = CustomDataset(data=test_data, transform=transform, data_folder=data_folder,
                                             type_set_folder='test', attack_type='lsb_classic/interleaving_squares',
                                           params=params)

    test_set_classic_sequential = CustomDataset(data=test_data, transform=transform, data_folder=data_folder,
                                                type_set_folder='test', attack_type='lsb_classic/sequential',
                                           params=params)

    test_set_oceanlotus_rows = CustomDataset(data=test_data, transform=transform, data_folder=data_folder,
                                             type_set_folder='test', attack_type='lsb_oceanlotus/interleaving_rows',
                                           params=params)

    test_set_oceanlotus_squares = CustomDataset(data=test_data, transform=transform, data_folder=data_folder,
                                                type_set_folder='test', attack_type='lsb_oceanlotus/interleaving_squares',
                                           params=params)

    test_set_oceanlotus_sequential = CustomDataset(data=test_data, transform=transform, data_folder=data_folder,
                                                   type_set_folder='test', attack_type='lsb_oceanlotus/sequential',
                                           params=params)

    train_set = torch.utils.data.ConcatDataset([train_set_classic_rows, train_set_classic_squares,
                                                train_set_classic_sequential,
                                                train_set_oceanlotus_rows, train_set_oceanlotus_squares,
                                                train_set_oceanlotus_sequential])

    val_set = torch.utils.data.ConcatDataset([val_set_classic_rows, val_set_classic_squares,
                                              val_set_classic_sequential, val_set_oceanlotus_rows,
                                              val_set_oceanlotus_squares, val_set_oceanlotus_sequential])


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

    test_loader_classic_rows_loader = torch.utils.data.DataLoader(test_set_classic_rows, batch_size=batch_size)
    test_loader_classic_squares_loader = torch.utils.data.DataLoader(test_set_classic_squares, batch_size=batch_size)
    test_loader_classic_sequential_loader = torch.utils.data.DataLoader(test_set_classic_sequential, batch_size=batch_size)
    test_loader_oceanlotus_rows_loader = torch.utils.data.DataLoader(test_set_oceanlotus_rows, batch_size=batch_size)
    test_loader_oceanlotus_squares_loader = torch.utils.data.DataLoader(test_set_oceanlotus_squares, batch_size=batch_size)
    test_loader_oceanlotus_sequential_loader = torch.utils.data.DataLoader(test_set_oceanlotus_sequential,
                                                                        batch_size=batch_size)

    return (train_loader, val_loader, test_loader_classic_rows_loader, test_loader_classic_squares_loader,
            test_loader_classic_sequential_loader, test_loader_oceanlotus_rows_loader,
            test_loader_oceanlotus_squares_loader, test_loader_oceanlotus_sequential_loader)
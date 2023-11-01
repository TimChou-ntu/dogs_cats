from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import os

class CatDogDataset(Dataset):
    def __init__(self, dir, mode='train', transform = None):
        self.dir = dir
        self.dir = dir
        self.mode= mode
        if transform is None:
            # Data augmentation, TODO: add more augmentations
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ColorJitter(),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(128),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

        self.file_list = os.listdir(dir)
           
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(os.path.join(self.dir, img_path))
        if self.transform:
            img = self.transform(img)
        if self.mode == 'train':
            if 'dog' in img_path:
                label = 1
            else:
                label = 0
            img = img.numpy()
            return img.astype('float32'), label
        else:
            img = img.numpy()
            return img.astype('float32'), img_path
        

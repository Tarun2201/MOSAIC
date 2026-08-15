import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class TrainDataset(Dataset):
    def __init__(self, dataframe, patch_size, config, mode='train', client_id=None):
        """
        config: {
            'dataset': 'brats',
            'task': 'binary' or 'multiclass',
            'combine': {
                'core': ['necrosis', 'enhancing'],
                'edema': ['edema']
            },
            'clients': {
                "1": ["flair"],
                "2": ["flair", "t1ce"],
                ...
            }
        }
        client_id: int, which client this dataset is for
        """
        self.dataframe = dataframe
        self.config = config
        self.patch_size = patch_size
        self.client_id = client_id

        # Get modalities for this client from config
        modalities = config['clients'][str(client_id)]
        # Deduplicate while preserving order
        seen = set()
        self.modalities = []
        for m in modalities:
            if m not in seen:
                seen.add(m)
                self.modalities.append(m)

        print("Client", client_id, "using modalities:", self.modalities)

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(patch_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(patch_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.dataframe)

    def _load_image(self, index):
        img_name_flair = self.dataframe.iloc[index, 0]
        
        # Path mapping for all modalities
        path_map = {
            'flair': img_name_flair,
            't1ce': img_name_flair.replace("flair", "t1ce"),
            't2': img_name_flair.replace("flair", "t2")
        }
        
        # Load only the modalities for this client
        channels = []
        for mod in self.modalities:
            img_path = path_map[mod]
            channels.append(Image.open(img_path).convert("L"))
        
        # Merge channels: 1 -> (ch, ch, ch), 2 -> (ch1, ch2, ch2), 3 -> as-is
        if len(channels) == 1:
            image = Image.merge("RGB", (channels[0], channels[0], channels[0]))
        elif len(channels) == 2:
            image = Image.merge("RGB", (channels[0], channels[1], channels[1]))
        else:  # 3 channels
            image = Image.merge("RGB", (channels[0], channels[1], channels[2]))

        return image

    def _process_labels(self, index):
        if self.config["task"] == "binary":
            return torch.tensor(self.dataframe.iloc[index, 2], dtype=torch.long).unsqueeze(0) # direct binary label
        else:
            label_dict = self.dataframe.iloc[index, 3:].to_dict()  # after 'label' column
            combined_labels = {}
            for new_class, old_classes in self.config["combine"].items():
                combined_labels[new_class] = int(any(label_dict.get(cls, 0) for cls in old_classes))

            keys = sorted(self.config['combine'].keys())
            return torch.tensor([combined_labels[k] for k in keys], dtype=torch.long)


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = self._load_image(index)
        label = self._process_labels(index)

        pos_1 = self.transform(image)
        pos_2 = self.transform(image)

        return pos_1, pos_2, label


class ImageDataset(Dataset):
    def __init__(self, dataframe, patch_size, config, mode='train', client_id=None):
        self.dataframe = dataframe
        self.config = config
        self.mode = mode
        self.client_id = client_id

        # Get modalities for this client
        modalities = config['clients'][str(client_id)]
        seen = set()
        self.modalities = []
        for m in modalities:
            if m not in seen:
                seen.add(m)
                self.modalities.append(m)
        print("Client", client_id, "using modalities:", self.modalities)
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(patch_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(90),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(patch_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.dataframe)

    def _load_image(self, index):
        img_name_flair = self.dataframe.iloc[index, 0]
        
        path_map = {
            'flair': img_name_flair,
            't1ce': img_name_flair.replace("flair", "t1ce"),
            't2': img_name_flair.replace("flair", "t2")
        }
        
        channels = []
        for mod in self.modalities:
            img_path = path_map[mod]
            channels.append(Image.open(img_path).convert("L"))
        
        if len(channels) == 1:
            image = Image.merge("RGB", (channels[0], channels[0], channels[0]))
        elif len(channels) == 2:
            image = Image.merge("RGB", (channels[0], channels[1], channels[1]))
        else:
            image = Image.merge("RGB", (channels[0], channels[1], channels[2]))

        return image

    def _process_labels(self, index):
        if self.config["task"] == "binary":
            return torch.tensor(self.dataframe.iloc[index, 2],dtype=torch.long).unsqueeze(0)  # direct binary label
        else:
            label_dict = self.dataframe.iloc[index, 3:].to_dict()
            combined_labels = {}
            for new_class, old_classes in self.config["combine"].items():
                combined_labels[new_class] = int(any(label_dict.get(cls, 0) for cls in old_classes))

            
            keys = sorted(self.config['combine'].keys())
            return torch.tensor([combined_labels[k] for k in keys], dtype=torch.long)


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = self._load_image(index)
        label = self._process_labels(index)
        transformed = self.transform(image)
        num_modalities = len(self.modalities)
        transformed = transformed[:num_modalities]

        return transformed, label
    

class InferenceDataset(Dataset):
    def __init__(self, dataframe, patch_size, config, client_id=None):
        self.dataframe = dataframe
        self.config = config
        self.client_id = client_id

        # Get modalities for this client
        modalities = config['clients'][str(client_id)]
        seen = set()
        self.modalities = []
        for m in modalities:
            if m not in seen:
                seen.add(m)
                self.modalities.append(m)
    
        self.transform = transforms.Compose([
                transforms.CenterCrop(patch_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.dataframe)

    def _load_image(self, index):
        img_name_flair = self.dataframe.iloc[index, 0]
        img_name = img_name_flair.split('/')[-1]
        
        path_map = {
            'flair': img_name_flair,
            't1ce': img_name_flair.replace("flair", "t1ce"),
            't2': img_name_flair.replace("flair", "t2")
        }
        
        channels = []
        for mod in self.modalities:
            img_path = path_map[mod]
            channels.append(Image.open(img_path).convert("L"))
        
        if len(channels) == 1:
            image = Image.merge("RGB", (channels[0], channels[0], channels[0]))
        elif len(channels) == 2:
            image = Image.merge("RGB", (channels[0], channels[1], channels[1]))
        else:
            image = Image.merge("RGB", (channels[0], channels[1], channels[2]))
        
        img_name_seg = img_name_flair.replace('flair', 'seg')
        seg = Image.open(img_name_seg).convert("RGB")

        return img_name, image, seg

    def _process_labels(self, index):
        if self.config["task"] == "binary":
            return torch.tensor(self.dataframe.iloc[index, 2],dtype=torch.long).unsqueeze(0)  # direct binary label
        else:
            label_dict = self.dataframe.iloc[index, 3:].to_dict()
            combined_labels = {}
            for new_class, old_classes in self.config["combine"].items():
                combined_labels[new_class] = int(any(label_dict.get(cls, 0) for cls in old_classes))

            
            keys = sorted(self.config['combine'].keys())
            return torch.tensor([combined_labels[k] for k in keys], dtype=torch.long)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name, image, seg = self._load_image(index)
        # Check unique pixel values per channel to detect a blank slice
        unique_values = set()
        has_zero_channel = False
        for channel in image.split():
            channel_data = set(channel.getdata())
            unique_values.update(channel_data)
            if len(channel_data) <= 1:
                has_zero_channel = True
        seg = self.transform(seg)
        transformed = self.transform(image)
        num_modalities = len(self.modalities)
        transformed = transformed[:num_modalities]
        if len(unique_values) <= 1 or has_zero_channel:
            return "blank", img_name, transformed, seg
        return "data", img_name, transformed, seg
import os
import random
import pickle
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image
from utils.general import load_image


class Dataset(data.Dataset):
    def __init__(self,
        data_dir,
        transform,
        views=6,
        face_parser=None):

        self.data_dir = data_dir
        self.views = views

        self.subjects = os.listdir(self.data_dir)
        print(f"Found {len(self.subjects)} subjects in the folder: {self.data_dir}")

        self.transform = transform
        self.face_parser = face_parser

    def __getitem__(self, index):
        '''
        Returns
        ----------
        org_images: segmented original image tensor of shape (num_real_views,3,H,W)
        images: segmented transformed image tensor of shape (num_views,3,H,W)
        subject_name: name of the subject (folder)
        '''
        subject_name = self.subjects[index]
        subject_dir = os.path.join(self.data_dir, subject_name)

        org_imgs = []
        input_imgs = []
        for image_file in os.listdir(subject_dir):
            # only select first self.views images
            if len(org_imgs) == self.views:
                break
            if (image_file.endswith(".png")):
                image_path = os.path.join(self.data_dir, subject_name, image_file)
                print(image_path)
                image = load_image(image_path) # (H,W,3)
                img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) # (1, 3, H, W)
                org_imgs.append(img_tensor)
        org_imgs = torch.cat(org_imgs[:self.views], dim=0) # (num_views, 3, H, W)
        
        # segment face
        face_mask = self.face_parser.face_mask_without_eyes_mouth(org_imgs) # (num_view, H, W)
        org_imgs = org_imgs * face_mask.unsqueeze(1)
        
        # tranform image
        input_imgs = [self.transform(org_imgs[i]).unsqueeze(0) for i in range(org_imgs.shape[0])]

        # if the the number of images are less than requirement views, 
        # repeat the last image for remaining times
        if len(input_imgs) < self.views:
            input_imgs.extend([input_imgs[-1]]*(self.views-len(org_imgs)))
        input_imgs = torch.cat(input_imgs[:self.views], dim=0)  # (N,3,H,W)

        output = {
            'org_images': org_imgs,  # (num_real_views, 3, h, w)
            'images': input_imgs,  # (num_view, 3, h, w)
            'subject_name': subject_name
        }
        return output

    def __len__(self):
        return len(self.subjects)


def get_dataloader(config, data_dir, drop_last=False):
    data_config = config.data
    augment_config = config.augment

    from torchvision import transforms
    from dataset.transform import Normalize
    from dataset.face_segment import FaceParser
    transform = transforms.Compose([
        Normalize(
            tuple(augment_config.normalize.mean),
            tuple(augment_config.normalize.std)
        )
    ])
    dataset = Dataset(
        data_dir=data_dir,
        transform=transform,
        views=data_config.views,
        face_parser=FaceParser(config.device)
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=drop_last
    )
    return data_loader

if __name__ == "__main__":
    import cv2
    from utils.general import load_config

    config_path = 'conf/inference.yaml'
    config = load_config(config_path)

    data_dir = config.data.data_dir
    dataloader = get_dataloader(config, data_dir)
    dataset = dataloader.dataset
    print(len(dataset))

    data_0 = dataset[0]
    print(data_0['org_images'].shape)
    print(data_0['images'].shape)

    cv2.imshow((data_0['org_images'][0].permute(1, 2, 0).numpy()[...,::-1]*255))
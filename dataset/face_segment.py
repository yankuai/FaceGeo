import facer
from enum import Enum
import torch

class FaceRegion(Enum):
    BACKGROUND = 0
    FACE = 1
    LEFT_EYE_BROW = 2
    RIGHT_EYE_BROW = 3
    LEFT_EYE = 4
    RIGHT_EYE = 5
    NOSE = 6
    UPPER_LIP = 7
    MOUTH_INTERIOR = 8
    LOWER_LIP = 9
    HAIR = 10

class FaceParser():
    def __init__(self, device):
        self.device = device
        self.face_detector = facer.face_detector('retinaface/mobilenet', device=device)
        self.face_parser = facer.face_parser('farl/lapa/448', device=device)

    def get_face_mask(self, images):
        with torch.inference_mode():
            faces = self.face_detector(images)
        with torch.inference_mode():
            faces = self.face_parser(images, faces)

        seg_logits = faces['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w

        return seg_probs

    def face_mask_without_eyes_mouth(self, images):
        """
        Parameters
        ----------
        images: tensor [B, C, H, W], values range in [0, 1]
        """
        images = (images*255).to(torch.uint8).to(self.device)
        seg_probs = self.get_face_mask(images)
        seg_probs = seg_probs[:, [1, 2, 3, 6, 7, 9], ...].sum(dim=1).clamp(max=1).int()
        return seg_probs.cpu()

if __name__ == "__main__":
    # test face parser
    import os
    import numpy as np
    import cv2
    from PIL import Image
    from google.colab.patches import cv2_imshow

    face_parser = FaceParser("cpu")
    img_path = "input/S1/v1.png"
    image = np.asarray(Image.open(img_path).convert('RGB')).astype('float32')/255 # 0-1, RGB
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
    face_mask = face_parser.face_mask_without_eyes_mouth(img_tensor) # (1, H, W)

    img_tensor = img_tensor * face_mask
    cv2.imshow(img_tensor[0].permute(1, 2, 0).numpy()[...,::-1]*255)

def load_config(path):
    import torch
    import yaml
    from munch import DefaultMunch

    with open(path) as f:
        dataMap = yaml.safe_load(f)
    config = DefaultMunch.fromDict(dataMap)

    if not torch.cuda.is_available():
        config.device = 'cpu'
    
    return config

def load_image(path, type='RGB'):
    import numpy as np
    from PIL import Image

    if type == 'RGB':
        return np.asarray(Image.open(path).convert('RGB')).astype('float32') / 255
    elif type == 'L':
        return np.asarray(Image.open(path)).astype('float32') / 255
import os
import torch
import numpy as np

from utils.general import load_config 
from utils.mesh import load_priors, pos_map_to_vertex, save_obj
from dataset.subject_dataset import get_dataloader
from model.detr_model import DETRModel

class Batch():
    def __init__(self, sample, uv_prior=None, can_pos_map=None, device='cuda'):
        self.org_imgs = sample['org_images'].to(device)
        self.input = sample['images'].to(device)

        self.uv_prior = uv_prior.to(device)
        self.can_pos_map = can_pos_map.to(device)

def eval(config, out_dir, dataloader, model, uv_prior, can_pos_map, pos_mask, faces, uv_map):
    model.eval()

    bs = config.data.batch_size
    with torch.no_grad():
        for sample in dataloader:
            batch = Batch(sample, uv_prior=uv_prior, can_pos_map=can_pos_map, device=config.device)

            pos_map = model(batch)
            pos_map = pos_map * pos_mask

            # covnert pos map to verts
            pred_verts = pos_map_to_vertex(pos_map.permute(0, 2, 3, 1), uv_map)

            # save mesh
            for i in range(pred_verts.shape[0]):
                save_path = os.path.join(out_dir, f'{sample["subject_name"][i]}.obj')
                save_obj(save_path, pred_verts[i].detach().cpu().numpy(), faces)

def main(config):
    data_dir = config.data.data_dir
    dataloader = get_dataloader(config, data_dir)
    print(f"Inference on {len(dataloader.dataset)} subjects.")

    model = DETRModel(config).to(config.device)
    checkpoint_path = config.model.checkpoint
    print(f'Loading checkpoint from:{checkpoint_path}')
    state_dict = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(state_dict)

    faces, pos_mask, can_pos_map, uv_prior, uv_map = load_priors(config.prior_path)

    out_dir = config.out_dir
    os.makedirs(out_dir, exist_ok=True)
    eval(config, out_dir, dataloader, model, uv_prior, can_pos_map, pos_mask, faces, uv_map)

    return

if __name__ == "__main__":
    config_path = 'conf/inference.yaml'
    config = load_config(config_path)
    
    main(config)
    exit(0)
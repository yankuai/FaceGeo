import os
import numpy as np

def load_obj(filename):
    import re

    def fromregex119(file, regexp, dtype, encoding=None):
        content = file.read()
        seq = regexp.findall(content)
        output = np.array(seq, dtype=dtype)
        return output

    _vertex_regex = re.compile("^v\s+(\S+)\s+(\S+)\s+(\S+)", re.MULTILINE)
    _face_regex = re.compile("^f\s+(\S+)\s+(\S+)\s+(\S+)", re.MULTILINE)

    # CHECK THAT .OBJ EXISTS
    if not os.path.isfile(filename):
        raise ValueError("OBJ file not found : "+filename)

    # read vertex and faces
    verts = fromregex119(open(filename), _vertex_regex, np.dtype(np.float64))
    tris = fromregex119(open(filename), _face_regex, np.dtype(np.int64))
    # decrease 1 to all array elements (change to zero base)
    tris = tris-1

    return {'verts':verts, 'tris':tris}

def save_obj(filename, verts, tris):
    with open(filename, 'w') as f:
        header  = np.array(['MV OBJ'])
        np.savetxt(f, header, fmt="# %s")
        np.savetxt(f, verts, fmt="v %f %f %f")
        np.savetxt(f, tris+1, fmt="f %d %d %d")

def load_priors(dir_path):
    import torch
    import cv2
    from utils.general import load_image
    
    # load faces tensor
    template_faces_obj_path = os.path.join(dir_path, 'template_face.obj')
    faces = torch.tensor(load_obj(template_faces_obj_path)['tris'], dtype=torch.int32).cpu().numpy()
    
    # load position map mask
    # the output position map's resolution is 256
    reso = 256
    mask_path = os.path.join(dir_path, 'face_eyes_mask.png')
    pos_mask = load_image(mask_path, type='L')
    pos_mask = torch.from_numpy(cv2.resize(pos_mask, (reso, reso)).astype('int'))[None, None,:,:]

    # load canonical position map and uv prior
    def get_uv_prior(canonical_pos_path, canonical_tex_path):
        # canonical_pos_map needs normalize
        canonical_pos_map = torch.tensor(np.load(canonical_pos_path)['pos']).permute(0, 3, 1, 2)  # (1, 3, 256, 256)
        canonical_pos_map = (canonical_pos_map - canonical_pos_map.min()) / (canonical_pos_map.max() - canonical_pos_map.min())
        # canonical_tex_map is already normalized
        canonical_tex_map = torch.tensor(np.load(canonical_tex_path)['pos'])[..., :2].permute(0, 3, 1, 2)  # (1, 2, 256, 256)
        return canonical_pos_map, canonical_tex_map

    canonical_pos_path = os.path.join(dir_path, 'pos_canonical_space.npz')
    canonical_tex_path = os.path.join(dir_path, 'tex_canonical_space.npz')
    can_pos_map, canonical_tex_map = get_uv_prior(canonical_pos_path, canonical_tex_path)
    uv_prior = torch.cat((canonical_tex_map, can_pos_map), dim=1)  # (1, 5, 256, 256)

    # load uv map
    uv_map = np.load(os.path.join(dir_path, 'uvs.npz'))    
    uv_map = torch.from_numpy(uv_map["uvs"]).float()

    return faces, pos_mask, can_pos_map, uv_prior, uv_map

def pos_map_to_vertex(pos_map, uv_map):
    import torch

    def get_normalized_uv_map(uv_map):
        new_map = torch.zeros_like(uv_map)
        new_map[:, 0] = uv_map[:, 0]
        new_map[:, 1] = 1 - uv_map[:, 1]
        grid = (2 * new_map - 1)
        return grid

    def transfer_texture_to_vertex_attribute(texture, uv_map, mode = 'bilinear', padding = 'border', align_corners = False):
        '''
        transfer a texture [N, w, h, 3] to per vertex color given a uvmap
        Args:
            texture: tensor of shape [B, 3, w, h], B is the batch
            uv_map: [N, 2] N being the vertices number
        Returns:
            a pytorch tensor of per vertex color [B, N, 3]
        '''
        assert(texture.dim() == 4 and uv_map.dim() == 2 and uv_map.shape[-1] == 2 )
        output = torch.nn.functional.grid_sample(texture, uv_map.unsqueeze(0).unsqueeze(0).repeat(texture.shape[0], 1, 1, 1), mode = mode, padding_mode = padding, align_corners = align_corners)

        colors = output.permute(0, 3, 1, 2).squeeze(-1)
        return colors

    grid = get_normalized_uv_map(uv_map)
    reconstructed = transfer_texture_to_vertex_attribute(pos_map.permute(0, 3, 1, 2), grid.to(pos_map.device))
    return reconstructed
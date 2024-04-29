from model.encoder.resnet import Resnet34_16s, Resnet18_16s
from model.decoder.unet import UNet
from model.transformer.DETR_transformer import TransformerDecoder, TransformerDecoderLayer
import torch
import torch.nn as nn
from einops import rearrange

class DETRModel(nn.Module):

    def __init__(self, config):
        super(DETRModel, self).__init__()
        self.config = config
        self.views = config.data.views

        self.encoder_dim = 128
        # image_size after resize is 256, stride is 16
        # encoder_output_size = image_size / stride
        self.encoder_output_size = 16

        # Encoders
        self.encoder_img = Resnet34_16s(num_classes=self.encoder_dim, pretrained=True)
        # UV input is the concatenation of canonical position map with 3 
        # channels and canonical texture map with 2 channels, so it has 5 
        # channels
        self.encoder_uv = Resnet18_16s(input_dim=5, out_dim=self.encoder_dim)

        # dim after concat
        self.d_model = self.encoder_dim
        self.total_view = self.views + 1

        # Self- and Cross-attention layer to fuse the multi-view image features,
        # use the DETR decoder layer 
        decoder_layer = TransformerDecoderLayer(
            d_model=self.d_model, 
            nhead=8, 
            dim_feedforward=2048, 
            dropout=0.1, 
            activation="relu", 
            normalize_before=True
        )
        decoder_norm = nn.LayerNorm(self.d_model)
        self.cross_attn_layers = TransformerDecoder(
            decoder_layer, 
            num_layers=6, 
            norm=decoder_norm
        )
        self._reset_parameters(self.cross_attn_layers.parameters())

        # positional encoding is the addition of
        # - learnable patch embedding: shared by patches of the same idx in all image views 
        # - view embedding: shared by patches of the same image view
        
        # For the patch embedding
        self.patch_embed = nn.Parameter(
            torch.randn(1, self.encoder_output_size * self.encoder_output_size, 
            self.d_model)
        )

        # For the view embedding
        self.pose_head = nn.Sequential(
            nn.Conv2d(self.encoder_dim, self.encoder_dim, 1), 
            nn.BatchNorm2d(self.encoder_dim), 
            nn.ReLU()
        )
        self.pos_linear = nn.Linear(self.encoder_dim, self.encoder_dim)

        # Decoder
        decoder_chanels = self.d_model
        self.up = nn.functional.upsample_bilinear
        # output channel number is 3 representing the (x,y,z) channel of 
        # the position map 
        self.decoder = UNet(n_channels=decoder_chanels, n_classes=3, bilinear=False)
        
    def _reset_parameters(self, parameters):
        for p in parameters:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch):
        imgs, uv_prior, can_pos_map = batch.input, batch.uv_prior, batch.can_pos_map

        input_spatial_dim = imgs.size()[3:]
        bs, vn, ic, ih, iw = imgs.shape
        imgs = imgs.view(-1, ic, ih, iw) # (B*V, C, H, W)

        # image encoder
        feat_img = self.encoder_img(imgs) # (B*V, 512, size, size)

        # extract view embeddings from image features
        view_embed = self.pose_head(feat_img.clone())
        view_embed = nn.functional.avg_pool2d(view_embed, view_embed.size()[3])
        view_embed = view_embed.view(view_embed.size(0), -1)
        view_embed = self.pos_linear(view_embed) # (B*V, 512)
        view_embed = view_embed.view(bs, vn, -1)

        feat_img = rearrange(feat_img, '(b v) c h w -> b v c h w', b=bs, v=vn) # (B, V, 512, size, size)
        size = self.encoder_output_size

        # add the view_embed and patch_embed as the positional embedding
        src = rearrange(feat_img, 'b v c h w -> b v (h w) c')  # (B, V!!!, size*size, 512)
        src += view_embed[:, :, None, :]
        src += self.patch_embed[:, None, :, :]
        src = rearrange(src, 'b v (h w) c -> (v h w) b c', v=vn, h=size, w=size)  # (V*size*size, B, 512)
            
        # extract uv feature
        feat_uv = self.encoder_uv(uv_prior).repeat(bs, 1, 1, 1)  # (B, 512, size, size)

        # go through transformer decoder layers with self- and cross-attention
        # to fuse the image features queried by the uv features
        tgt = rearrange(feat_uv, 'b c h w -> (h w) b c', h=size, w=size)
        src = self.cross_attn_layers(tgt, src, query_pos=None)[0]
        feat = rearrange(src, '(h w) b c -> b c h w', b=bs, c=self.d_model, h=size, w=size)
        
        # go through UNet decoder
        feat = self.up(feat, size=input_spatial_dim)
        pred_pos = self.decoder(feat)

        # the output of decoder is the displacement based on input canonical
        # position map
        pred_pos = pred_pos + can_pos_map.repeat(bs, 1, 1, 1)
        return pred_pos

def main(config):
    from train_gan import Batch
    model = DETRModel(config)
    input = torch.zeros((2, config.data.views, 3, config.data.resolution, config.data.resolution))
    uv_map = torch.zeros((1, 5, config.data.resolution, config.data.resolution))
    can_pos = torch.zeros((1, 3, config.data.resolution, config.data.resolution))
    batch = {'images': input}
    batch = Batch(batch, uv_map=uv_map, can_pos_map=can_pos, device='cpu')
    output = model(batch)[-1]
    print(output.shape) # (B, 3, 256, 256)

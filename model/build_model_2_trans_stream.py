import timm
import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from model.model_block import Attn2AttnBlock_2_Trans_Stream, up_sampling_by_convt, up_sampling_by_interpolate, ASPP, BinCenterPred
from encoder.pvt import pvt_v2_b5

class Encoder(nn.Module):
    def __init__(self, pretrained=None):
        super(Encoder, self).__init__()
        self.backbone = pvt_v2_b5()
        self.dim_list = self.backbone.dim_list[::-1]

    def forward(self, x):
        return self.backbone(x)[::-1]

    def freeze_layer(self, enable=False):
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class Decoder(nn.Module):
    def __init__(self, args, channel_list):
        super(Decoder, self).__init__()
        self.channel_list = channel_list

        embed_dims_list = [512, 320, 128, 64]
        num_heads_list = [8, 4, 2, 1]
        mlp_ratio_list = [4, 4, 4, 8]
        window_size_list = [4, 4, 7, 14]

        self.aspp = ASPP(self.channel_list[0], self.channel_list[0])
        self.attn_32 = Attn2AttnBlock_2_Trans_Stream(
            in_channels=self.channel_list[0], embed_dims=embed_dims_list[0], v_dims=self.channel_list[0],
            patch_size=window_size_list[0], num_heads=num_heads_list[0], mlp_ratio=mlp_ratio_list[0]
        )
        self.attn_16 = Attn2AttnBlock_2_Trans_Stream(
            in_channels=self.channel_list[1] + embed_dims_list[0] // 2, embed_dims=embed_dims_list[1],
            v_dims=self.channel_list[1] + embed_dims_list[0] // 2, patch_size=window_size_list[1],
            num_heads=num_heads_list[1], mlp_ratio=mlp_ratio_list[1]
        )
        self.attn_8 = Attn2AttnBlock_2_Trans_Stream(
            in_channels=self.channel_list[2] + embed_dims_list[1] // 2, embed_dims=embed_dims_list[2],
            v_dims=self.channel_list[2] + embed_dims_list[1] // 2, patch_size=window_size_list[2],
            num_heads=num_heads_list[2], mlp_ratio=mlp_ratio_list[2]
        )
        self.attn_4 = Attn2AttnBlock_2_Trans_Stream(
            in_channels=self.channel_list[3] + embed_dims_list[2] // 2, embed_dims=embed_dims_list[3],
            v_dims=self.channel_list[3] + embed_dims_list[2] // 2, patch_size=window_size_list[3],
            num_heads=num_heads_list[3], mlp_ratio=mlp_ratio_list[3]
        )

        self.up_32 = up_sampling_by_convt(in_channels=embed_dims_list[0], out_channels=embed_dims_list[0] // 2)
        self.up_16 = up_sampling_by_convt(in_channels=embed_dims_list[1], out_channels=embed_dims_list[1] // 2)
        self.up_8 = up_sampling_by_convt(in_channels=embed_dims_list[2], out_channels=embed_dims_list[2] // 2)

    def forward(self, feat_maps):
        feat_map_32, feat_map_16, feat_map_8, feat_map_4 = feat_maps
        feat_map_32 = self.aspp(feat_map_32)
        attn_32 = self.attn_32(feat_map_32)

        attn_32_up = self.up_32(attn_32)
        feat_map_16 = torch.cat([feat_map_16, attn_32_up], dim=1)
        attn_16 = self.attn_16(feat_map_16)

        attn_16_up = self.up_16(attn_16)
        feat_map_8 = torch.cat([feat_map_8, attn_16_up], dim=1)
        attn_8 = self.attn_8(feat_map_8)

        attn_8_up = self.up_8(attn_8)
        feat_map_4 = torch.cat([feat_map_4, attn_8_up], dim=1)
        attn_4 = self.attn_4(feat_map_4)
        return attn_4, attn_8, attn_16, attn_32

class Attn2AttnDepth(nn.Module):
    def __init__(self, args):
        super(Attn2AttnDepth, self).__init__()
        pretrained = args.pretrained
        if args.dataset == 'kitti':
            self.args = args.kitti
        else:
            self.args = args.nyu
        self.encoder = Encoder(pretrained=pretrained)
        self.channels_list = self.encoder.dim_list
        # print(self.channels_list)
        self.decoder = Decoder(self.args, self.channels_list)
        self.bin_pred = BinCenterPred(in_features=512, out_features=64,
                                      max_depth=self.args.max_depth, min_depth=self.args.min_depth)

        self.apply(self._init_weights)
        print('init model parameters...')

        self.encoder.backbone.init_weights(pretrained)
        print('init extractor by {}'.format(pretrained))

    def forward(self, x):
        feat_maps = self.encoder(x)
        decoder_out = self.decoder(feat_maps)
        bin_centers = self.bin_pred(decoder_out[-1])
        out = torch.softmax(decoder_out[0], dim=1)
        out = torch.sum(out * bin_centers, dim=1, keepdim=True)
        out = up_sampling_by_interpolate(out, scale_factor=4)
        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # def train(self, mode=True):
    #     super().train(mode)
    #     self.encoder.freeze_layer()


if __name__ == '__main__':
    '''
    torch.Size([1, 64, 88, 280])
    torch.Size([1, 128, 44, 140])
    torch.Size([1, 320, 22, 70])
    torch.Size([1, 512, 11, 35])
    '''
    from utils.opt import args
    from torchsummary import summary

    input = torch.randn((1, 3, 480, 640))
    m = Attn2AttnDepth(args)
    # summary(m, (3, 352, 1120))
    out = m(input)
    print(out.shape)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.attn_block import WindowAttentionBlock, CrossWindowAttentionBlock, Attention

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

def up_sampling_by_convt(in_channels, out_channels, kernel_size=3, stride=2,
                         padding=1, output_padding=1, bn=True, relu=True):
    '''
    使用nn.ConvTranspose2d()实现的上采样方法，如果需要实现一倍的上采样，则需要设置stride=2
    :param in_channels: 特征图的输入通道数
    :param out_channels: 输出通道数
    :param kernel_size: 卷积核尺寸
    :param stride: 步长，如果要实现一倍的上采样，则stride=2
    :param padding: 输入的时候对特征图的填充
    :param output_padding: 输出特征图的填充
    :param bn: 是否使用bn
    :param relu: 是否使用relu激活
    :return: 返回上采样后的特征图
    '''
    layers = []
    bias = True if bn == False else False
    layers.append(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)
    )
    if bn == True:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu == True:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def up_sampling_by_interpolate(input, scale_factor=2):
    return F.interpolate(input, scale_factor=scale_factor)

class AsppBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1):
        super(AsppBlock, self).__init__()
        self.padding = dilation if dilation > 1 else 0

        self.blk = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=dilation,
                      padding=self.padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.blk(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rate=1):
        super(ASPP, self).__init__()

        self.blk1 = AsppBlock(in_channels, out_channels, kernel_size=1, stride=1, dilation=rate)
        self.blk2 = AsppBlock(in_channels, out_channels, kernel_size=3, stride=1, dilation=rate * 6)
        self.blk3 = AsppBlock(in_channels, out_channels, kernel_size=3, stride=1, dilation=rate * 12)
        self.blk4 = AsppBlock(in_channels, out_channels, kernel_size=3, stride=1, dilation=rate * 18)

        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.act5 = nn.GELU()

        self.cat = AsppBlock(out_channels * 5, out_channels, kernel_size=1, stride=1)
        self.cbam = DyCAConv(inp=out_channels * 5, oup=out_channels * 5, kernel_size=3, stride=1)

    def forward(self, x):
        b, c, h, w = x.size()

        conv1 = self.blk1(x)
        conv2 = self.blk2(x)
        conv3 = self.blk3(x)
        conv4 = self.blk4(x)

        global_feat = torch.mean(x, 2, keepdim=True)
        global_feat = torch.mean(global_feat, 3, keepdim=True)
        global_feat = self.act5(self.conv5(global_feat))
        global_feat = F.interpolate(global_feat, (h, w), None, mode='bilinear', align_corners=True)

        feat_cat = torch.cat([conv1, conv2, conv3, conv4, global_feat], dim=1)
        cbam = self.cbam(feat_cat)
        out = self.cat(cbam)
        return out

class MultiPatchAspp(nn.Module):
    def __init__(self, in_channels, patch_kernels=[5, 7, 9], embed_dims=[256, 192, 64]):
        super(MultiPatchAspp, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_dims[0], kernel_size=patch_kernels[0], padding=autopad(patch_kernels[0]))
        self.conv2 = nn.Conv2d(in_channels, embed_dims[1], kernel_size=patch_kernels[1], padding=autopad(patch_kernels[1]))
        self.conv3 = nn.Conv2d(in_channels, embed_dims[2], kernel_size=patch_kernels[2], padding=autopad(patch_kernels[2]))

        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        # self.attn1 = Attention(embed_dims[0], num_heads=4)
        # self.attn2 = Attention(embed_dims[1], num_heads=4)
        # self.attn3 = Attention(embed_dims[2], num_heads=8)

    def forward(self, x):
        b, c, h, w = x.shape
        patch_conv1 = self.conv1(x).flatten(start_dim=2).transpose(1, 2).contiguous()
        patch_conv1 = self.norm1(patch_conv1)
        # patch_attn1 = self.attn1(patch_conv1, h, w)
        patch_conv1 = patch_conv1.reshape(b, -1, h, w)

        patch_conv2 = self.conv2(x).flatten(start_dim=2).transpose(1, 2).contiguous()
        patch_conv2 = self.norm2(patch_conv2)
        # patch_attn2 = self.attn2(patch_conv2, h, w)
        patch_conv2 = patch_conv2.reshape(b, -1, h, w)

        patch_conv3 = self.conv3(x).flatten(start_dim=2).transpose(1, 2).contiguous()
        patch_conv3 = self.norm3(patch_conv3)
        # patch_attn3 = self.attn3(patch_conv3, h, w)
        patch_conv3 = patch_conv3.reshape(b, -1, h, w)
        # print(f'patch_attn1 shape: {patch_attn1.shape}, patch_attn2 shape: {patch_attn2.shape}, patch_attn1 shape: {patch_attn3.shape}')
        return torch.cat([patch_conv1, patch_conv2, patch_conv3], dim=1)

class MultiPatchAttention(nn.Module):
    def __init__(self, in_channels, patch_kernels=[5, 7, 9], embed_dims=[256, 192, 64]):
        super(MultiPatchAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_dims[0], kernel_size=patch_kernels[0],
                               padding=autopad(patch_kernels[0]))
        self.conv2 = nn.Conv2d(in_channels, embed_dims[1], kernel_size=patch_kernels[1],
                               padding=autopad(patch_kernels[1]))
        self.conv3 = nn.Conv2d(in_channels, embed_dims[2], kernel_size=patch_kernels[2],
                               padding=autopad(patch_kernels[2]))

        self.patch_attn1 = WindowAttentionBlock(embed_dims[0], embed_dims[0], num_heads=4)
        self.patch_attn2 = WindowAttentionBlock(embed_dims[1], embed_dims[1], num_heads=4)
        self.patch_attn3 = WindowAttentionBlock(embed_dims[2], embed_dims[2], num_heads=8)

    def forward(self, x):
        patch_conv1 = self.conv1(x)
        patch_conv2 = self.conv2(x)
        patch_conv3 = self.conv3(x)

        patch_attn1 = self.patch_attn1(patch_conv1)
        patch_attn2 = self.patch_attn2(patch_conv2)
        patch_attn3 = self.patch_attn3(patch_conv3)

        return torch.cat([patch_attn1, patch_attn2, patch_attn3], dim=1)


class DyCAConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, reduction=16):
        super(DyCAConv, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, kernel_size, padding=kernel_size // 2, stride=stride),
                                  nn.BatchNorm2d(oup),
                                  nn.SiLU())

        self.dynamic_weight_fc = nn.Sequential(
            nn.Linear(inp, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Compute dynamic weights
        x_avg_pool = nn.AdaptiveAvgPool2d(1)(x)
        x_avg_pool = x_avg_pool.view(x.size(0), -1)
        dynamic_weights = self.dynamic_weight_fc(x_avg_pool)

        out = identity * (dynamic_weights[:, 0].view(-1, 1, 1, 1) * a_w +
                          dynamic_weights[:, 1].view(-1, 1, 1, 1) * a_h)

        return self.conv(out)

class CBAMLayer(nn.Module):
    def __init__(self, in_channels, reductions=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reductions, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels // reductions, in_channels, kernel_size=1, bias=False)
        )

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=spatial_kernel, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = x * channel_out
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = x * spatial_out
        return x

class CBRBlock(nn.Module):
    '''
    一个标准的conv + bn + relu的模块
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(CBRBlock, self).__init__()
        self.blk1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.blk1(x)

class CSConvBlock(nn.Module):
    '''
    一个普通的卷积模块（conv + bn + relu）后面跟了一个CBAM注意力
    '''
    def __init__(self, in_channels, kernel_size=3, stride=1, reductions=16):
        super(CSConvBlock, self).__init__()
        self.cbr = CBRBlock(in_channels, in_channels, kernel_size, stride, padding=1)
        self.channel_spatial_attn = DyCAConv(in_channels, in_channels, kernel_size, stride, reductions)

    def forward(self, x):
        res = x
        x = self.cbr(x)
        x = self.channel_spatial_attn(x)
        # x = F.relu(x)
        out = x + res
        return out

class Attn2AttnBlock(nn.Module):
    def __init__(self, in_channels, embed_dims, patch_size, v_dims, num_heads, mlp_ratio, qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Attn2AttnBlock, self).__init__()
        self.conv_stream = CSConvBlock(in_channels)
        self.attn_stream = WindowAttentionBlock(
            input_dim=in_channels, embed_dim=embed_dims, window_size=patch_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path
        )
        self.cross_attn_stream = CrossWindowAttentionBlock(
            input_dim=in_channels, embed_dim=embed_dims, v_dim=embed_dims, window_size=patch_size, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path
        )

    def forward(self, x):
        cnn_out = self.conv_stream(x)  # in_chan -> in_chan
        # print('cnn_out shape', cnn_out.shape)
        attn_out = self.attn_stream(x)  # in_chan -> embed_dims
        # print('attn_out shape:', attn_out.shape)
        out = self.cross_attn_stream(cnn_out, attn_out)
        return out

class Attn2AttnBlock_Cnn_Only(nn.Module):
    def __init__(self, in_channels, embed_dims, patch_size, v_dims, num_heads, mlp_ratio, qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Attn2AttnBlock_Cnn_Only, self).__init__()
        self.conv_stream = CSConvBlock(in_channels)
        self.conv = nn.Conv2d(in_channels, embed_dims, kernel_size=3, padding=1, bias=False, stride=1)

    def forward(self, x):
        cnn_out = self.conv_stream(x)
        out = self.conv(cnn_out)
        return out

class Attn2AttnBlock_Trans_Only(nn.Module):
    def __init__(self, in_channels, embed_dims, patch_size, v_dims, num_heads, mlp_ratio, qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Attn2AttnBlock_Trans_Only, self).__init__()
        self.attn_stream = WindowAttentionBlock(
            input_dim=in_channels, embed_dim=embed_dims, window_size=patch_size, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path
        )

    def forward(self, x):
        attn_out = self.attn_stream(x)
        return attn_out
    
class Attn2AttnBlock_2_Cnn_Stream(nn.Module):
    def __init__(self, in_channels, embed_dims, patch_size, v_dims, num_heads, mlp_ratio, qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Attn2AttnBlock_2_Cnn_Stream, self).__init__()
        self.conv_stream_1 = CSConvBlock(in_channels)
        self.conv_stream_2 = CSConvBlock(in_channels)
        self.conv = nn.Conv2d(in_channels, embed_dims, kernel_size=3, padding=1, bias=False, stride=1)
        self.cross_attn = CrossWindowAttentionBlock(
            input_dim=in_channels, embed_dim=embed_dims, v_dim=embed_dims, window_size=patch_size, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path
        )

    def forward(self, x):
        cnn_out = self.conv_stream_1(x)
        cnn_out_attn = self.conv_stream_2(x)
        conv_stead_of_attn = self.conv(cnn_out_attn)
        out = self.cross_attn(cnn_out, conv_stead_of_attn)
        return out
    
class Attn2AttnBlock_2_Trans_Stream(nn.Module):
    def __init__(self, in_channels, embed_dims, patch_size, v_dims, num_heads, mlp_ratio, qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Attn2AttnBlock_2_Trans_Stream, self).__init__()
        self.attn_stream_1 = WindowAttentionBlock(
            input_dim=in_channels, embed_dim=in_channels, window_size=patch_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path
        )
        self.attn_stream_2 = WindowAttentionBlock(
            input_dim=in_channels, embed_dim=embed_dims, window_size=patch_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path
        )
        self.cross_attn = CrossWindowAttentionBlock(
            input_dim=in_channels, embed_dim=embed_dims, v_dim=embed_dims, window_size=patch_size, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path
        )

    def forward(self, x):
        attn_stead_of_cnn = self.attn_stream_1(x)
        attn_stream = self.attn_stream_2(x)
        cross_attn = self.cross_attn(attn_stead_of_cnn, attn_stream)
        return cross_attn
        
class BinCenterPred(nn.Module):
    def __init__(self, max_depth, min_depth, in_features, out_features, act=nn.GELU):
        super(BinCenterPred, self).__init__()
        hid_features = in_features * 4
        self.act = act()
        self.fc1 = nn.Linear(in_features, hid_features)
        self.fc2 = nn.Linear(hid_features, out_features)
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, x):
        # [n, c, h, w] -> [n, c, h*w] -> [n, c, 1]
        x = torch.mean(x.flatten(start_dim=2), dim=2)
        x = self.act(self.fc1(x))
        x = self.fc2(x)   # [n, out_feat]
        # 求出每个输出通道上的概率分布
        bins = torch.softmax(x, dim=1)
        bins = bins / bins.sum(dim=1, keepdim=True)  # 每个通道分布概率占总概率，但是bins.sum(dim=1) = 1，这不是多此一举么？
        bin_widths = (self.max_depth - self.min_depth) * bins
        bin_widths = F.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.contiguous().view(n, dout, 1, 1)
        return centers


if __name__ == '__main__':
    input = torch.randn((1, 512, 11, 22))
    # m_conv = CSConvBlock(in_channels=2048)
    # m_attn = AttentionBlock(feat_size=20, in_channels=2048, embed_dims=512, patch_size=1)
    # m_cross = CrossAttentionBlock(feat_size=20, in_channels_q=2048, in_channels_kv=512, embed_dims=512, v_dims=512, patch_size=1)
    #
    # m_c_out = m_conv(input)
    # m_a_out = m_attn(input)
    # cross_out = m_cross(m_c_out, m_a_out)
    # print(m_c_out.shape)
    # print(m_a_out.shape)
    # print(cross_out.shape)
    m = MultiPatchAttention(512)
    out = m(input)
    print(out.shape)


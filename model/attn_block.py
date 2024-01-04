import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from utils.tool import window_partition, window_reverse

class Mlp(nn.Module):
    '''
    一个简单的mlp网络，有两层nn.Linear()构成
    '''
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class PatchEmbed(nn.Module):
    '''
    和pvt的代码一样，作用是将输入图像编码成一个个展平后的特征 -> [b, num_patches, c]
    返回值是：[b, num_patches, embed_dim], (h, w)，h = ori_h//patch_size[0], w = ori_w//patch_size[1]
    '''
    def __init__(self, img_size, patch_size, in_channels, embed_dim=768):
        super(PatchEmbed, self).__init__()
        '''
        img_size: [h, w]
        patch_size: [kernel_h, kernel_w]
        '''
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size

        # 确保输入图像能够被patch_kernel整除
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W   # 一张图片一共能够被编码成多少个小的Patch

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape

        # [b, c, h, w] -> [b, ebmed_dim, h//patch[0], w//patch[1]] -> [b, embed_dim, num_patches] -> [b, num_patches, embed_dim]
        x = self.proj(x).flatten(start_dim=2).transpose(1, 2).contiguous()
        x = self.norm(x)

        h, w = h // self.patch_size[0], w // self.patch_size[1]
        return x, (h, w)

class Attention(nn.Module):
    '''
    输入：[b, n, c]
    输出：[b, n, c]
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(Attention, self).__init__()
        # 因为是多头注意力，所以需要确保输入维度能够被num_heads整除
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # 如果sr_ratio > 1，则会执行下面的代码，会使用sr_ration大小的卷积核，配合对应大小的步长，将特征图进一步缩小[b, n, c]中的n变小
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        # [b, n, c] -> [b, n, num_heads, c//num_heads] -> [b, num_heads, n, c//num_heads]
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, h, w).contiguous()  # [b, n, c] -> [b, c, n] -> [b, c, h, w]
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1).contiguous() # [b, c, h, w] -> [b, c, h*w] -> [b, h*w, c]
            x_ = self.norm(x_)
            # [b, h*w, c] -> [b, h*w, c*2] -> [b, h*w, 2, num_heads, c//num_heads] -> [2, b, num_heads, h*w, c//num_heads]
            kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]  # k, v维度一样都是[b, num_heads, h*w, c//num_heads]

        # [b, num_heads, n, c//num_heads] @ [b, num_heads, c//num_heads, h*w] -> [b, num_heads, h*w, h*w]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # [b, num_heads, h*w, h*w] @ [b, num_heads, h*w, c//num_heads] -> [b, num_heads, h*w, c//num_heads]
        # -> [b, h*w, num_heads, c//num_heads] -> [b, n, c]
        x = (attn @ v).transpose(1, 2).reshape(b, n, c).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.dim = dim
        window_size = to_2tuple(window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, n, c = x.shape

        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, v_dim, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(CrossWindowAttention, self).__init__()
        self.dim = dim
        window_size = to_2tuple(window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_positon_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # [2, win_size, win_size]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)  # [2, win_size, win_size] -> [2, win_size * win_size]
        relative_coords = coords_flatten[..., None] - coords_flatten[:, None, ...]  # 先维度扩充，再相减，运用了广播机制
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[..., 0] += self.window_size[0] - 1
        relative_coords[..., 1] += self.window_size[1] - 1
        relative_coords[..., 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(v_dim, v_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_positon_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, v):
        b, n, c = x.shape   # [b*h'*w', win_size*win_size, c], b' = b*h'*w', n=win_size*win_size
        # [b, n, dim] -> [b, n, num_heads, dim//num_heads] -> [b, num_heads, n, dim//num_heads]
        q = self.q(x).reshape(b, n, self.num_heads, -1).transpose(1, 2).contiguous()
        # [b, n, dim] -> [b, n, 2, num_heads, dim//num_heads] -> [2, b, num_heads, n, dim//num_heads]
        kv = self.kv(v).reshape(b, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]
        q = q * self.scale

        # [b, num_heads, n, dim//num_heads] @ [b, num_heads, dim//num_heads, n] -> [b, num_heads, n, n]
        attn = q @ k.transpose(-2, -1).contiguous()
        relative_position_bias = self.relative_positon_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)

        # [b, num_heads, n, n] @ [b, num_heads, n, dim//num_heads] -> b, num_heads, n, dim//num_heads] -> [b, n, dim]
        x = (attn @ v).transpose(1, 2).reshape(b, n, c).contiguous()
        x = self.proj(x)  # [b, n, dim] -> [b, n, v_dim]
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    '''
    交叉注意力计算
    可以接受两个不同的特征图作为输入，维度是[b, n, c]
    假设两组输入向量的维度不同，分别是[b, n, c_x], [b, n, c_v]，经过运算后返回的维度是[b, n, c_v]
    '''
    def __init__(self, dim, num_heads=8, v_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(CrossAttention, self).__init__()
        assert v_dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, v_dim, bias=qkv_bias)
        self.kv = nn.Linear(v_dim, v_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(v_dim, v_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_kv = nn.Conv2d(v_dim, v_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_kv = nn.LayerNorm(v_dim)

    def forward(self, x, v, h, w):
        b, n, c = x.shape
        b_v, n_v, c_v = v.shape
        # [b, n, c] -> [b, n, num_heads, c_v//num_heads] -> [b, num_heads, n, c_v//num_heads]
        q = self.q(x).reshape(b, n, self.num_heads, c_v // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            # [b_v, n_v, c_v] -> [b_v, c_v, n_v] -> [b_v, c_v, h, w]
            kv_ = v.permute(0, 2, 1).reshape(b_v, c_v, h, w).contiguous()
            # [b_v, c_v, h, w] -> [b_v, c_v, h//sr_ratio, w//sr_ratio] -> [b_v, c_v, h'*w'] -> [b_v, h'*w', c_v]
            kv_ = self.sr_kv(kv_).reshape(b_v, c_v, -1).permute(0, 2, 1).contiguous()
            kv_ = self.norm_kv(kv_)
            # [b_v, h'*w', c_v] -> [b_v, h'*w', c_v*2] -> [b_v, h'*w', 2, num_heads, c_v//num_heads] -> [2, b_v, num_heads, h'*w', c_v//num_heads]
            kv = self.kv(kv_).reshape(b_v, -1, 2, self.num_heads, c_v // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(v).reshape(b_v, -1, 2, self.num_heads, c_v // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]  # [b_v, num_heads, h'*w', c_v//num_heads]

        # [b, num_heads, n, c_v//num_heads] @ [b, num_heads, c_v//num_heads, n'] -> [b, num_heads, n, n']
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # [b, num_heads, n, n'] @ [b, num_heads, n', c_v//num_heads] -> [b, num_heads, n, c_v//num_heads] -> [b, n, num_heads, c_v//num_heads]
        # -> [b, n, c_v]
        x = (attn @ v).transpose(1, 2).reshape(b, n, c_v).contiguous()
        x = self.proj(x) # c_v -> c_v
        x = self.proj_drop(x)
        return x   # [b, n, c_v]

class CrossWindowAttentionBlock(nn.Module):
    def __init__(self, input_dim, embed_dim, v_dim, window_size, num_heads=8, mlp_ratio=4, norm_layer=nn.LayerNorm,
                 qkv_bias=False, qk_scale=None, drop=0., drop_path=0., attn_drop=0.):
        super(CrossWindowAttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        if input_dim != embed_dim:
            self.proj_q = nn.Conv2d(input_dim, embed_dim, kernel_size=3, padding=1)
        else:
            self.proj_q = None

        if v_dim != embed_dim:
            self.proj_v = nn.Conv2d(v_dim, embed_dim, kernel_size=3, padding=1)
        else:
            self.proj_v = None

        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.v_dim = embed_dim
        self.norm_1 = norm_layer(embed_dim)
        self.norm_2 = norm_layer(embed_dim)
        self.norm_3 = norm_layer(self.v_dim)
        self.norm_4 = norm_layer(embed_dim)
        self.attn = CrossWindowAttention(dim=embed_dim, window_size=window_size, num_heads=num_heads, v_dim=self.v_dim,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        mlp_hidden_dim = int(self.v_dim * mlp_ratio)
        self.mlp = Mlp(in_features=self.v_dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, v):
        # 这两个if，将x, v的维度都统一到embed_dim上
        if self.proj_q is not None:
            x = self.proj_q(x)
        if self.proj_v is not None:
            v = self.proj_v(v)
        x_proj = x   # 最后输出的残差连接
        v_proj = v

        b, c, h, w = x.size()
        # n = h * w
        x = x.flatten(2).transpose(1, 2).contiguous()  # [b, c, h*w] -> [b, h*w, c]
        v = v.flatten(2).transpose(1, 2).contiguous()

        x = self.norm_1(x)
        x = x.reshape(b, h, w, c)

        v = self.norm_2(v)
        v = v.reshape(b, h, w, c)

        # 需要让输入的特征图尺寸能够被window_size整除，所以要对特征图进行填充
        pad_l, pad_t = 0, 0
        # 比如h = w = 4, win_size = 7，需要填充3
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        v = F.pad(v, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, h_pad, w_pad, _ = x.shape

        # [b, h, w, c] -> [b * h' * w', win_size, win_size, c]  h' = h_p // win_size
        x_window = window_partition(x, self.window_size)
        x_window = x_window.view(-1, self.window_size * self.window_size, c)  # -> [b*h'*w', win_size*win_size, c]

        v_window = window_partition(v, self.window_size)
        v_window = v_window.view(-1, self.window_size * self.window_size, v_window.shape[-1])

        attn_window = self.attn(x_window, v_window)   # [b, n, v_dim] -> 实际上还是[b*h'w', win_size*win_size, v_dim]
        # -> [b*h'*w', win_size, win_size, v_dim]
        attn_window = attn_window.view(-1, self.window_size, self.window_size, self.v_dim)
        x = window_reverse(attn_window, self.window_size, h_pad, w_pad)  # [b, h, w, v_dim]

        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, ...].contiguous()
        x = x.view(b, h * w, self.v_dim)
        x = self.drop_path(x)
        x = self.norm_3(x)
        x = self.mlp(x)
        x = x + self.drop_path(x)

        x = self.norm_4(x)
        x = x.view(-1, h, w, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        return x + x_proj + v_proj

class WindowAttentionBlock(nn.Module):
    def __init__(self, input_dim, embed_dim, window_size=7, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super(WindowAttentionBlock, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        if input_dim != embed_dim:
            self.proj_q = nn.Conv2d(input_dim, embed_dim, kernel_size=3, padding=1)
        else:
            self.proj_q = None

        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.norm_1 = norm_layer(embed_dim)
        self.norm_2 = norm_layer(embed_dim)
        self.norm_3 = norm_layer(embed_dim)
        self.attn = WindowAttention(dim=embed_dim, window_size=window_size, num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        if self.proj_q is not None:
            x = self.proj_q(x)
        b, c, h, w = x.size()
        x = x.flatten(2).transpose(1, 2).contiguous() # [b, em_dim, h, w] -> [b, em_dim, h*w] -> [b, h*w, em_dim]
        x = self.norm_1(x)
        x = x.view(b, h, w, c)
        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, h_pad, w_pad, _ = x.shape

        x_windows = window_partition(x, self.window_size)  # [b, h, w, em_dim] -> [b*h'*w', win_size, win_size, em_dim], h'=h//win_size
        # -> [b*h'*w', win_size, win_size, c]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(x_windows) # [b, n, em_dim] -> 实际上还是[b*h'w', win_size*win_size, em_dim]
        # -> [b*h'*w', win_size, win_size, v_dim]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        x = window_reverse(attn_windows, self.window_size, h_pad, w_pad)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, ...].contiguous()
        x = x.view(b, h * w, self.embed_dim)
        x = self.drop_path(x)
        x = self.norm_2(x)
        x = self.mlp(x)
        x = x + self.drop_path(x)
        x = self.norm_3(x)
        x = x.view(-1, h, w, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        return x

class AttentionBlock(nn.Module):
    '''
    一个self-attn模块
    输入：[n, c, h, w]维度的特征图，经过patch_embed -> pos_embed -> self-attn -> mlp -> out
    输出：[n, c, h', w']特征图, h' = h // patch_size, w' = w // patch_size
    训练的时候可传入的参数：
    embed_dims, patch_size, num_heads, mlp_ratio, sr_ratio
    '''
    def __init__(self, feat_size, in_channels, embed_dims, patch_size=16, num_heads=8, mlp_ratio=4, qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(AttentionBlock, self).__init__()
        patch_embed = PatchEmbed(feat_size, patch_size, in_channels, embed_dims)
        num_patches = patch_embed.num_patches
        pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
        pos_drop = nn.Dropout(drop)

        self.norm1 = norm_layer(embed_dims)
        self.attn = Attention(embed_dims, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(embed_dims)
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dims, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        setattr(self, 'patch_embed', patch_embed)
        setattr(self, 'pos_embed', pos_embed)
        setattr(self, 'pos_drop', pos_drop)

        trunc_normal_(pos_embed, std=.02)

    def forward(self, x):
        b = x.shape[0]
        patch_embed = getattr(self, 'patch_embed')
        pos_embed = getattr(self, 'pos_embed')
        pos_drop = getattr(self, 'pos_drop')
        x, (h, w) = patch_embed(x)
        # x = pos_drop(x + pos_embed)
        # 这里做了一个小的残差连接
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return x

class CrossAttentionBlock(nn.Module):
    '''
    一个cross-attn模块
    输入: x[b, c1, h, w], v[b, c2, h, w]
    输出: [b, c2, h', w']  h' = h // patch_size, w' = w // patch_size
    '''
    def __init__(self, feat_size, in_channels_q, in_channels_kv, embed_dims, patch_size=16, num_heads=8, v_dims=None,
                 mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.1, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1):
        super(CrossAttentionBlock, self).__init__()
        patch_embed_q = PatchEmbed(feat_size, patch_size, in_channels_q, embed_dims)
        patch_embed_kv = PatchEmbed(feat_size, patch_size, in_channels_kv, v_dims)
        num_patches = patch_embed_kv.num_patches

        pos_embed_q = nn.Parameter(torch.zeros(num_patches, embed_dims))
        pos_embed_kv = nn.Parameter(torch.zeros(num_patches, v_dims))
        pos_drop = nn.Dropout(drop)

        self.norm1_q = norm_layer(embed_dims)
        self.norm1_kv = norm_layer(v_dims)
        self.cross_attn = CrossAttention(embed_dims, num_heads, v_dims, qkv_bias, qk_scale, attn_drop,
                                         proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(v_dims)
        mlp_hidden_dim = int(v_dims * mlp_ratio)
        self.mlp = Mlp(in_features=v_dims, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        setattr(self, 'patch_embed_q', patch_embed_q)
        setattr(self, 'patch_embed_kv', patch_embed_kv)
        setattr(self, 'pos_embed_q', pos_embed_q)
        setattr(self, 'pos_embed_kv', pos_embed_kv)
        setattr(self, 'pos_drop', pos_drop)

        trunc_normal_(pos_embed_q, std=.02)
        trunc_normal_(pos_embed_kv, std=.02)

    def forward(self, x, v):
        b = x.shape[0]
        patch_embed_q = getattr(self, 'patch_embed_q')
        patch_embed_kv = getattr(self, 'patch_embed_kv')
        pos_embed_q = getattr(self, 'pos_embed_q')
        pos_embed_kv = getattr(self, 'pos_embed_kv')
        pos_drop = getattr(self, 'pos_drop')

        x, (h, w) = patch_embed_q(x)
        v, _ = patch_embed_kv(v)
        # x = pos_drop(x + pos_embed_q)
        # v = pos_drop(v + pos_embed_kv)
        x = v + self.drop_path(self.cross_attn(self.norm1_q(x), self.norm1_kv(v), h, w))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return x


if __name__ == '__main__':
    # x = torch.randn((1, 169, 32))
    # v = torch.randn((1, 169, 128))
    # m = CrossAttention(dim=32, v_dim=128, sr_ratio=4)
    # out = m(x, v, 13, 13)
    # print(out.shape)

    # x = torch.randn((1, 3, 96, 128))
    # m = AttentionBlock(feat_size=(96, 128), in_channels=3, embed_dims=64, patch_size=2, sr_ratio=4)
    # out = m(x)
    # print(out.shape)

    x = torch.randn((1, 2048, 20, 20))
    v = torch.randn((1, 2048, 20, 20))
    m = WindowAttentionBlock(input_dim=2048, embed_dim=512, window_size=7)
    out = m(x)
    print(out.shape)




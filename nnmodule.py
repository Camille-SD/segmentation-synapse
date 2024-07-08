'''Modules used by the models'''
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch
import timm

# UNETR

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2


        self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(
                x
            )  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)

        return x


class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
                2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
           q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(
                1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x
    
    def get_attention(self, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
                2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
           q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)

        #return torch.min(attn, dim=1)[0]
        return attn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(
                x
        ) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, out_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, out_features)

        return x


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
                dim,
                n_heads=n_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=dim,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x
    
    def get_attn(self, x):
        return self.attn.get_attention(self.norm1(x))
        


class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size=384,
            patch_size=16,
            in_chans=3,
            n_classes=1000,
            embed_dim=768,
            depth=12,
            n_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
                torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        # self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        features = []

        cls_token = self.cls_token.expand(
                n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
            features.append(x.clone())

        # x = self.norm(x)

        # cls_token_final = x[:, 0]  # just the CLS token
        # x = self.head(cls_token_final)

        return features

    def attention_map(self, x):
        '''Returns a list of attention maps for each block'''
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        attentions = []

        cls_token = self.cls_token.expand(
                n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
            attn_x = block.get_attn(x).mean(dim=1)
            attentions.append(attn_x.clone().detach().cpu().numpy()) # dim : (n_blocks, n_samples, n_patches + 1, n_patches + 1)
        
        return torch.Tensor(np.array(attentions)).permute(1, 0, 2, 3)
    
    def attention_rollout(self, x, discard_ratio=0.8, head_fusion='mean'):
        '''Returns a list of all attention rollouts for each block. 
        Attention rollout is the dot product between the attention map of the current layer and the attention map of the previous layers.
        The alogirthm is simple :
        if i == 0:
            rollouts[i] = attention_maps[i]
        else:
            rollout[i] = attention_maps[i] @ rollouts[i-1]'''
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        attention_maps = []

        cls_token = self.cls_token.expand(
                n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)
        
        # How much of the lowest attention values is dropped
        for block in self.blocks:
            x = block(x)
            attn_x = block.get_attn(x) # dim : (n_samples, n_heads, n_patches + 1, n_patches + 1)
            attention_maps.append(attn_x.clone().detach().cpu().numpy()) 
        
        attention_maps = torch.Tensor(np.array(attention_maps))
        print(attention_maps.shape) # dim : (n_blocks, n_heads, n_patches + 1, n_patches + 1)
        result = torch.eye(attention_maps[0].shape[-1], device=attention_maps[0].device)

        result_list = []

        for attn_x in attention_maps:
            if head_fusion == 'mean':
                attention_heads_fused = attn_x.mean(dim=1) # dim : (n_samples, n_patches + 1, n_patches + 1)
            elif head_fusion == 'max':
                attention_heads_fused = attn_x.max(dim=1)[0] # dim : (n_samples, n_patches + 1, n_patches + 1)
            elif head_fusion == 'min':
                attention_heads_fused = attn_x.min(dim=1)[0] # dim : (n_samples, n_patches + 1, n_patches + 1)
            else:
                raise "Attention head fusion type Not supported"
            flat = attention_heads_fused.view(attention_heads_fused.shape[0], attention_heads_fused.shape[1], -1) # dim : (n_samples, n_patches + 1, n_patches + 1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0
            
            I = torch.eye(attention_heads_fused.shape[-1], device=attention_heads_fused.device)
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
            result_list.append(result.clone())

        return result_list

class Deconv(nn.Module):
  # Used by UNETR
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )

  def forward(self, x):
    return self.conv(x)

class DoubleConv(nn.Module):
  # Used by UNETR
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )

  def forward(self, x):
    return self.conv(x)

# UNETR FUNCTION

def load_custom_model(model_custom, timm_name = "vit_base_patch16_224", depth = 12):

  model_official = timm.create_model(timm_name, pretrained=True)
  model_official.eval()

  model_custom.eval()

  for (n_o, p_o), (n_c, p_c) in zip(
        model_official.named_parameters(), model_custom.named_parameters()
):
    if 'block' not in n_o or int(n_o.split('.')[1]) < depth:
      p_c.data[:] = p_o.data
    else:
      break
  
  return model_custom


# U-NET

class DownSampleBlock(nn.Module):
    '''Reduce the dimension of the image in input by 2'''
    def __init__(self, in_channels, out_channels, dilation=1):
        super(DownSampleBlock, self).__init__()
        # We keep the same dimension in input and ouput
        self.conv = DoubleConvolution(in_channels, out_channels, dilation=dilation)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2))

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x), x

class DoubleConvolution(nn.Module):
    """Used by U-Net"""
    def __init__(self, in_channels, out_channels, dilation=1):
        super(DoubleConvolution, self).__init__()
        # We keep the same dimension in input and ouput

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=dilation, dilation=dilation)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
      x = self.relu(self.norm2(self.conv2(self.relu(self.norm1(self.conv1(x))))))
      return x

class UpSampleBlock(nn.Module):
    '''Increase the dimension of the input and reduce its number of channels
    out_channels reduces the number of channels of the input by 2 by default
    It can be set to a different value'''
    def __init__(self, in_channels, out_channels=None):
        super(UpSampleBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.up1 = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        
    def forward(self, x):
        x = self.up1(x)
        return x

class ResidualConnection(nn.Module):
    '''Concatenate inputs of two blocs'''

    def __init__(self, in_channels, out_channels):
        '''in_channels has the same dimensions as out_channels'''

        super(ResidualConnection, self).__init__()
        self.conv = DoubleConvolution(in_channels, out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

from nnmodule import PatchEmbed, Block, ResidualConnection, UpSampleBlock, DoubleConvolution, DownSampleBlock, Deconv, DoubleConv, VisionTransformer, load_custom_model
import torch.nn as nn
import torch
from rf_function import conv, upsample, downsample

class UNet(nn.Module):

    NB_OF_FILTERS = 64

    def __init__(self, input_size, num_classes:int=10, depth:int=2, dilation:int=1):
        '''### Initialize a UNet model
        input_size : dimension of input
        num_classes : specify the number of classes in ouput
        depth : the number of blocks (depth of the model)
        dilation : dilation rate of the convolutional layers (1 means basic convolution))'''
        super(UNet, self).__init__()

        in_channels = input_size[-1]
        channels = [in_channels, 64, 128, 256, 512, 1024, 2048]
        channels = channels[:depth + 2]
        print(channels)
        self.down_samples_blocks = nn.ModuleList([])
        self.up_samples_blocks = nn.ModuleList([])
        self.residual_connections = nn.ModuleList([])
        self.bottleneck = DoubleConvolution(in_channels=channels[-2], out_channels=channels[-1], dilation=dilation)
        self.segmentation_head = nn.Conv2d(in_channels=channels[1], out_channels=num_classes,
                                kernel_size=(3, 3), stride=(1, 1), padding=(1,1))
        for i in range(depth):
            self.down_samples_blocks.append(DownSampleBlock(in_channels=channels[i], out_channels=channels[i + 1], dilation=dilation))
            self.residual_connections.append(ResidualConnection(in_channels=channels[i + 2], out_channels=channels[i + 1]))
            self.up_samples_blocks.append(UpSampleBlock(in_channels=channels[i + 2], out_channels=channels[i + 1]))

    def forward(self, x):
        self.feature_maps = []
        # Encoder
        # Copy of output of each blocks before downsampling
        xs_down =[]
        for i, down_block in enumerate(self.down_samples_blocks):
            x, copy = down_block.forward(x)
            xs_down.append(copy)


        x = self.bottleneck.forward(x)
        xs_down = xs_down[::-1]
        # Decoder
        for i, (up_block, r_conn) in enumerate(zip(reversed(self.up_samples_blocks), reversed(self.residual_connections))):
            x_up = up_block.forward(x)
            x = r_conn(x_up, xs_down[i])
            
        x = self.segmentation_head(x)
        return x
    
    def get_receptive_field(self, dilation=1):
        '''Compute the receptive field of the model. The dilation used when creating the model should be specified'''
        rf = 1
        for up_block in self.up_samples_blocks:
            rf = upsample(rf)
        rf = conv(conv(rf, dilation=dilation), dilation=dilation)
        for down_block in self.down_samples_blocks:
            rf = downsample(rf, dilation=dilation)
        return rf

class UNETR(nn.Module):
  def __init__(self, img_size = 224, patch_size = 16, in_chans = 3, embed_dim = 768,
               depth = 12, n_heads = 12, mlp_ratio = 4, qkv_bias = True, pretrained_name = None,
               conv_init = True, skip_connections = [0], num_classes = 10):
    super().__init__()

    self.skip_connections = skip_connections
    self.patch_dim = img_size // patch_size
    self.conv_init = conv_init

    # Encoder

    self.vit = VisionTransformer(img_size = img_size,
                  patch_size = patch_size,
                  in_chans = in_chans,
                  embed_dim = embed_dim,
                  depth = depth,
                  n_heads = n_heads,
                  mlp_ratio = mlp_ratio,
                  qkv_bias = qkv_bias)
    if pretrained_name is not None:
      self.vit = load_custom_model(model_custom = self.vit, timm_name = pretrained_name, depth = depth)

    if conv_init:
      self.img_conv = DoubleConv(in_chans, 64)
    
    self.u11 = Deconv(embed_dim, 512)
    self.u12 = Deconv(512, 256)
    self.u13 = Deconv(256, 128)

    if len(self.skip_connections) > 1:
      self.u21 = Deconv(embed_dim, 512)
      self.u22 = Deconv(512, 256)

    if len(self.skip_connections) > 2:
      self.u31 = Deconv(embed_dim, 512)

    if len(self.skip_connections) > 3:
      self.u41 = nn.ConvTranspose2d(768, 512, 2, 2)

    # Decoder

    if len(self.skip_connections) == 1:
      self.dc1 = DoubleConv(128, 128)
      self.du1 = nn.ConvTranspose2d(128, 64, 2, 2)

    if len(self.skip_connections) == 2:
      self.dc2 = DoubleConv(256, 256)
      self.du2 = nn.ConvTranspose2d(256, 128, 2, 2)

      self.dc1 = DoubleConv(256, 128)
      self.du1 = nn.ConvTranspose2d(128, 64, 2, 2)

    if len(self.skip_connections) == 3:
      self.dc3 = DoubleConv(512, 512)
      self.du3 = nn.ConvTranspose2d(512, 256, 2, 2)

      self.dc2 = DoubleConv(512, 256)
      self.du2 = nn.ConvTranspose2d(256, 128, 2, 2)

      self.dc1 = DoubleConv(256, 128)
      self.du1 = nn.ConvTranspose2d(128, 64, 2, 2)
    
    if len(self.skip_connections) == 4:
      self.dc3 = DoubleConv(1024, 512)
      self.du3 = nn.ConvTranspose2d(512, 256, 2, 2)

      self.dc2 = DoubleConv(512, 256)
      self.du2 = nn.ConvTranspose2d(256, 128, 2, 2)

      self.dc1 = DoubleConv(256, 128)
      self.du1 = nn.ConvTranspose2d(128, 64, 2, 2)

    if self.conv_init:
      self.final_conv = DoubleConv(128, 64)
    else:
      self.final_conv = DoubleConv(64, 64)

    self.segmentation_head = nn.Conv2d(64, num_classes, 3, 1, 1)

  def forward(self, x):

    # Encoder

    if self.conv_init:
      conv_features = self.img_conv(x)
      # print(conv_features.shape)

    features = self.vit(x)

    n, _, c = features[0].shape
    features = [i[:, 1:, :].permute(0, 2, 1).reshape(n, c, self.patch_dim, self.patch_dim) for i in features]

    feature_0 = features[self.skip_connections[0]]
    feature_0 = self.u11(feature_0)
    feature_0 = self.u12(feature_0)
    feature_0 = self.u13(feature_0)

    # print(feature_0.shape)

    if len(self.skip_connections) > 1:
      feature_1 = features[self.skip_connections[1]]
      feature_1 = self.u21(feature_1)
      feature_1 = self.u22(feature_1)
      # print(feature_1.shape)

    if len(self.skip_connections) > 2:
      feature_2 = features[self.skip_connections[2]]
      feature_2 = self.u31(feature_2)    
      # print(feature_2.shape)

    if len(self.skip_connections) > 3:
      feature_3 = features[self.skip_connections[3]]
      feature_3 = self.u41(feature_3)
      # print(feature_3.shape)

    # Decoder

    if len(self.skip_connections) == 1:
      x1 = self.dc1(feature_0)
      x1 = self.du1(x1)

    if len(self.skip_connections) == 2:
      x2 = self.dc2(feature_1)
      x2 = self.du2(x2)

      x1 = torch.cat((feature_0, x2), 1)
      x1 = self.dc1(x1)
      x1 = self.du1(x1)

    if len(self.skip_connections) == 3:
      x3 = self.dc3(feature_2)
      x3 = self.du3(x3)

      x2 = torch.cat((feature_1, x3), 1)
      x2 = self.dc2(x2)
      x2 = self.du2(x2)

      x1 = torch.cat((feature_0, x2), 1)
      x1 = self.dc1(x1)
      x1 = self.du1(x1)
  
    if len(self.skip_connections) == 4:
      x3 = torch.cat((feature_2, feature_3), 1)
      x3 = self.dc3(x3)
      x3 = self.du3(x3)

      x2 = torch.cat((feature_1, x3), 1)
      x2 = self.dc2(x2)
      x2 = self.du2(x2)

      x1 = torch.cat((feature_0, x2), 1)
      x1 = self.dc1(x1)
      x1 = self.du1(x1)
    
    if self.conv_init:
      x1 = torch.cat((conv_features, x1), 1)
    x = self.final_conv(x1)
    out = self.segmentation_head(x)

    return out
  
  def attention_map(self, x):
    '''Returns a list of attention maps for each block'''
    return self.vit.attention_map(x)
  
  def attention_rollout_map(self, x, discard_ratio=0.8, head_fusion='mean'):
     '''Returns a list of attention rollout maps for each block'''
     return self.vit.attention_rollout(x, discard_ratio, head_fusion)

  if __name__ == "__main__":
     # Create UNET model

    model = UNet(input_size=(3, 224, 224), num_classes=10, depth=4, dilation=1)
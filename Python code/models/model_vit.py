"""
Compare the impact of different Attention mechanisms on CLEAR network performance
Using ViT-style global self-attention: divide the feature map into patches and perform global interaction between patch tokens
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

def complex_mse_loss(output, target):
    """
    output: Tensor of shape (batch_size,height, width, 2)
    target: Tensor of shape (batch_size,height, width, 2)
    """
    loss_real = F.mse_loss(output[..., 0], target[..., 0])
    loss_imag = F.mse_loss(output[..., 1], target[..., 1])
    return loss_real + loss_imag

class FFN_Module(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion_factor),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        return self.net(x)

class MHSA_Module(nn.Module):
    """ViT-style global self-attention with 8x8 patch embedding."""
    def __init__(self, dim, num_heads, dropout=0.1, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = patch_size * patch_size * dim
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.unpatch_embed = nn.Linear(dim, patch_dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, height, width, dim)
        batch_size, height, width, dim = x.shape
        patch_size = self.patch_size
        assert height % patch_size == 0 and width % patch_size == 0, (
            f"height ({height}) and width ({width}) must be divisible by patch_size ({patch_size})"
        )

        x_norm = self.layer_norm(x).contiguous()
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size

        # patchify: (batch, num_patches, patch_size*patch_size*dim)
        x_patches = x_norm.view(
            batch_size, num_patches_h, patch_size, num_patches_w, patch_size, dim
        )
        x_patches = x_patches.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_patches = x_patches.view(batch_size, num_patches_h * num_patches_w, -1)

        x_tokens = self.patch_embed(x_patches)
        attn_output, _ = self.self_attn(x_tokens, x_tokens, x_tokens)
        x_patches_out = self.unpatch_embed(attn_output)

        # unpatchify: (batch, height, width, dim)
        attn_output = x_patches_out.view(
            batch_size, num_patches_h, num_patches_w, patch_size, patch_size, dim
        )
        attn_output = attn_output.permute(0, 1, 3, 2, 4, 5).contiguous()
        attn_output = attn_output.view(batch_size, height, width, dim)
        return self.dropout(attn_output)

class Conv_Module(nn.Module):
    def __init__(self, dim, kernel_size=(7, 7), dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2), groups=dim)
        self.batch_norm = nn.BatchNorm2d(dim)
        self.pointwise_conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x_norm = self.layer_norm(x)
        x = x_norm.permute(0, 3, 1, 2)  # (batch, dim, height, width)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.permute(0, 2, 3, 1)       # (batch, height, width, dim)
        
        return self.dropout(x)


class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_expansion_factor=4, conv_kernel_size=(7,7), dropout=0.1):
        super().__init__()
        self.ff_module1 = FFN_Module(dim, ff_expansion_factor, dropout)
        self.mhsa_module = MHSA_Module(dim, num_heads, dropout)
        self.conv_module = Conv_Module(dim, conv_kernel_size, dropout)
        self.ff_module2 = FFN_Module(dim, ff_expansion_factor, dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = x + 0.5 * self.ff_module1(x)
        x = x + self.mhsa_module(x)
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ff_module2(x)

        return self.layer_norm(x)

class dc_layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, sampled, mask):

        batch_size,N1,N2,_ = input.shape
        _,N1_origin,N2_origin,_ = sampled.shape
        real_part = input[:,0:N1_origin,0:N2_origin,0]
        imag_part = input[:,0:N1_origin,0:N2_origin,1]
        complex_input = torch.view_as_complex(torch.stack((real_part, imag_part), dim=-1))
        inverse_fft = torch.fft.ifft2(complex_input)
        rl_inverse = torch.real(inverse_fft)
        im_inverse = torch.imag(inverse_fft)

        inverse_fft = torch.stack((rl_inverse,im_inverse), dim=-1)
        inverse_fft = (1-mask) * inverse_fft + sampled         

        inverse_fft = inverse_fft.contiguous()   
        x_data_consis = torch.fft.fft2(torch.view_as_complex(inverse_fft))   
        output = torch.stack((torch.real(x_data_consis), torch.imag(x_data_consis)), dim=-1)
        output_pad = torch.zeros((batch_size,N1,N2,2),dtype=torch.float32)
        output_pad[:,0:N1_origin,0:N2_origin,:] = output
        output_pad = output_pad.to(device=input.device)

        return output_pad


class CLEAR(nn.Module):
    def __init__(self, input_dim=2, model_dim=512, num_heads=8, num_layers=6, ff_expansion_factor=4, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.conformer_block1 = nn.ModuleList([
            ConformerBlock(model_dim, num_heads, ff_expansion_factor, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])
        self.conformer_block2 = nn.ModuleList([
            ConformerBlock(model_dim, num_heads, ff_expansion_factor, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(model_dim, input_dim)
        self.dc_layer = dc_layer()

    def forward(self, input, sampled, mask):
        """
        input  : Tensor of shape (batch_size,height, width, 2) where the last dimension represents real and imaginary parts
        sampled: Tensor of shape (batch_size,height, width, 2) representing the sampled time domain data
        mask   : Tensor of shape (batch_size,height, width, 2) representing the duplicated sampling mask along the last dimension 
                 to match the real and imaginary channels
        """

        x_projection = self.input_projection(input)  # (batch_size,height, width, model_dim)
        for conformer in self.conformer_block1:
            x_conformer1 = conformer(x_projection)
            x_projection = x_conformer1
        output_conformer1 = self.output_projection(x_conformer1)   # (batch_size,height, width, 2)
        output1 = self.dc_layer(output_conformer1, sampled, mask)    

        x_projection = self.input_projection(output1)
        for conformer in self.conformer_block2:
            x_conformer2 = conformer(x_projection)
            x_projection = x_conformer2
        output_conformer2 = self.output_projection(x_conformer2)   # (batch_size,height, width, 2)
        output2 = self.dc_layer(output_conformer2, sampled, mask)

        return output2
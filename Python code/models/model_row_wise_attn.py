"""
Compare the impact of different Attention mechanisms on CLEAR network performance
Using Row-Wise Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, height, width, dim = x.shape
        x_norm = self.layer_norm(x).contiguous()
        x_norm = x_norm.permute(0, 2, 1, 3).contiguous()           # (B, W, H, C)
        x_seq = x_norm.view(batch_size * width, height, dim)       # (B*W, H, C)
        attn_output, _ = self.self_attn(x_seq, x_seq, x_seq)
        attn_output = attn_output.view(batch_size, width, height, dim)  # (B, W, H, C)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()      # (B, H, W, C)
        return self.dropout(attn_output)

class CONV_Module(nn.Module):
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
        x = x.permute(0, 2, 3, 1)  # (batch, height, width, dim)
        
        return self.dropout(x)


class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_expansion_factor=4, conv_kernel_size=(7,7), dropout=0.1):
        super().__init__()
        self.ff_module1 = FFN_Module(dim, ff_expansion_factor, dropout)
        self.mhsa_module = MHSA_Module(dim, num_heads, dropout)
        self.conv_module = CONV_Module(dim, conv_kernel_size, dropout)
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

    def forward(self,input,sampled,mask):
        lambda_mul = 1e5
        real_part = input[:,:,:,0]
        imag_part = input[:,:,:,1]
        complex_input = torch.view_as_complex(torch.stack((real_part, imag_part), dim=-1))
        inverse_fft = torch.fft.ifft2(complex_input)
        rl_inverse = torch.real(inverse_fft)
        im_inverse = torch.imag(inverse_fft)

        inverse_fft = torch.stack((rl_inverse,im_inverse), dim=-1)
        inverse_fft = (1-mask) * inverse_fft + sampled            
        x_data_consis = torch.fft.fft2(torch.view_as_complex(inverse_fft))   
        output = torch.stack((torch.real(x_data_consis), torch.imag(x_data_consis)), dim=-1)

        return output


class CLEAR_Row_Wise_Attn(nn.Module):
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

    def forward(self, input, sampled,mask):
        """
        input  : Tensor of shape (batch_size,height, width, 2) where the last dimension represents real and imaginary parts
        sampled: Tensor of shape (batch_size,height, width, 2) representing the sampled time domain data
        mask   : Tensor of shape (batch_size,height, width, 2) representing the duplicated sampling mask along the last dimension 
                 to match the real and imaginary channels
        """

        x_projection = self.input_projection(input)  
        for conformer in self.conformer_block1:
            x_conformer1 = conformer(x_projection)
            x_projection = x_conformer1
        output_conformer1 = self.output_projection(x_conformer1)   
        output1 = self.dc_layer(output_conformer1,sampled,mask)    

        x_projection = self.input_projection(output1)
        for conformer in self.conformer_block2:
            x_conformer2 = conformer(x_projection)
            x_projection = x_conformer2
        output_conformer2 = self.output_projection(x_conformer2) 
        output2 = self.dc_layer(output_conformer2,sampled,mask)

        return output2
        

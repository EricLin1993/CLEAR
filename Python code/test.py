import torch
import nmrglue as ng
import numpy as np
import time
from models.model import CLEAR
from utils.nus_proc_func import nus_proc,complex2hyper

if __name__ == '__main__':

    # === Load the pre-trained model ===
    input_dim = 2   # hyperparameter of CLEAR                        
    model_dim = 32                                     
    num_heads = 2
    num_layers = 2                                
    conv_kernel_size = (17,17)                        
    dropout = 0.1
    loaded_model = CLEAR(input_dim = input_dim, model_dim = model_dim, num_heads = num_heads, num_layers = num_layers, ff_expansion_factor=4,conv_kernel_size = conv_kernel_size, dropout=dropout)
    loaded_model.load_state_dict(torch.load('./best_model/best_model.pth'))



    data_type = 'PSRP'  # 'A3DK08',  'Yfgj',  'PSRP'
    print("Is CUDA available: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    fidr_3d_nus_2c_pad, fidr_3d_nus_origin_2c, fr_nus_2c, fr_2c, fidi_3d_nus_2c_pad, fidi_3d_nus_origin_2c, fi_nus_2c, fi_2c,mask_2c = nus_proc(data_type)
    N1,N2,N3,_ = fr_nus_2c.shape
    N1_origin,N2_origin,_,_ = fidr_3d_nus_origin_2c.shape
    recon_result = np.zeros((N1,N2,N3),dtype = complex)
    iz1 = 1
    iz2 = N3
    start = time.perf_counter()
    for recon_step in ['Xr', 'Xi']:
        mask = torch.tensor(mask_2c, dtype=torch.float32)
        if recon_step == 'Xr':
            f_nus_2c = fr_nus_2c
            fid_3d_nus_2c = fidr_3d_nus_origin_2c
            f_2c = fr_2c
        elif recon_step == 'Xi':
            f_nus_2c = fi_nus_2c
            fid_3d_nus_2c = fidi_3d_nus_origin_2c
            f_2c = fi_2c
        for i in range (iz1,iz2):
            test_input     =      f_nus_2c[:,:,i,:]
            sampled_signal = fid_3d_nus_2c[:,:,i,:]
            test_input = test_input.reshape(1,N1,N2,2)
            sampled_signal = sampled_signal.reshape(1,N1_origin,N2_origin,2)
            test_input     = torch.tensor(test_input, dtype=torch.float32)  
            sampled_signal = torch.tensor(sampled_signal, dtype=torch.float32)
            test_input = test_input.to(device)
            sampled_signal = sampled_signal.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                output2 = loaded_model.forward(test_input,sampled_signal,mask=mask)
            f_pre = output2.detach().cpu().numpy().reshape(N1,N2,2)   # output of second phase       
            out = f_pre[:,:,0]+1j*f_pre[:,:,1]
            recon_result[:,:,i] = out.squeeze()
        end = time.perf_counter()
        recon_result_t = np.zeros((N1_origin,N2_origin,N3),dtype = complex)
        for i in range (iz1,iz2):
            recon_result_t[:,:,i] = np.fft.ifft2(recon_result[0:N1_origin,0:N2_origin,i].squeeze())
        if recon_step == 'Xr':
            recon_r = recon_result_t
        elif recon_step == 'Xi':
            recon_i = recon_result_t
    recon_hyper_path = f"./Recon_Results/{data_type}_recon.ft1"
    complex2hyper(recon_r,recon_i,data_type,recon_hyper_path)
    print(f'Reconstruction time: {(end-start):.2f}s\n')
    print(f"Reconstruction saved at: {recon_hyper_path} \n need to be processed by NMRPipe script (NMRPipe Code/proc_indirct.com) ...")
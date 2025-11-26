%% 2D Simulation Data Generation
clc,clear;
tic;
N1 = 64; N2=64; Npeak_min=5; Npeak_max=15; nusr_min=0.10; nusr_max=0.25; SNR_dB_min=10; SNR_dB_max=15; noise_flag=1;win_flag=0;
for i=1:4
    [fid_2d_2c,fid_2d_nus_2c,f_2d_2c,f_2d_nus_2c,mask_2c] = sim2d_gen(N1,N2,Npeak_min,Npeak_max,nusr_min,nusr_max,SNR_dB_min,SNR_dB_max,noise_flag,win_flag); 
    save(sprintf('YourFolder/Data_%dx%d_nus_%d-%d_pks_%d-%d/%d.mat',N1,N2,round(nusr_min*100),round(nusr_max*100),Npeak_min,Npeak_max,i),'fid_2d_2c','fid_2d_nus_2c','f_2d_2c','f_2d_nus_2c','mask_2c');
    
end
toc;
disp('Simulation Data Generation Finished!');


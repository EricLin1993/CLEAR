function [fid_2d_2c,fid_2d_nus_2c,f_2d_2c,f_2d_nus_2c,mask_2c] = sim2d_gen(N1,N2,Npeak_min,Npeak_max,nusr_min,nusr_max,SNR_dB_min,SNR_dB_max,noise_flag,win_flag)  % N1,N2,nusr_min,nusr_max,SNR_dB,noise_flag
%     N1 = 32;
%     N2 = 128;
%     nusr_min = 0.10;      % Min_sampling rate
%     nusr_max = 0.25;      % Max_sampling rate
%     SNR_dB_min = 10;      % signal_noise_ratio  10
%     SNR_dB_max = 15;
%     noise_flag = 1;       %  1:noisy 0: clean
%     win_flag  = 1;        %  1:with SinBell  0: without window
% 
%     Npeak_min = 5; 
%     Npeak_max = 15;  
    A_min = 0;    
    A_max = 1;

    SW2_min = 1500; 
    SW2_max = 3000;
    SW1_min = 1800;
    SW1_max = 5400;
 
    R1_min = 5;
    R1_max = 60;
    R2_min = 5;
    R2_max = 60;

    
    %%%%% generate one signal %%%%%
    Npeak = Npeak_min + round(rand(1)*(Npeak_max-Npeak_min));
    SNR_dB = SNR_dB_min +round(rand(1)*(SNR_dB_max-SNR_dB_min)); 

    SW1 = SW1_min + round(rand(1)*(SW1_max-SW1_min));
    SW2 = SW2_min + round(rand(1)*(SW2_max-SW2_min));
    R1 = R1_min + rand(1, Npeak)*(R1_max-R1_min);
    R2 = R2_min + rand(1, Npeak)*(R2_max-R2_min);
    A = A_min + rand(1, Npeak)*(A_max-A_min);

    Omg1 = sort(rand(1, Npeak)*SW1*2*pi);  
    Omg2 = sort(rand(1, Npeak))*SW2*2*pi;

    while min(diff(Omg1))/SW1/2/pi*N1 < 2 
        Omg1 = sort(rand(1, Npeak)*SW1*2*pi);
    end
    
    while min(diff(Omg2))/SW2/2/pi*N2 < 2  
        Omg2 = sort(rand(1, Npeak))*SW2*2*pi;
    end
    
    t1 = (0: N1-1)/SW1;
    t2 = (0: N2-1)/SW2;   
    fid_2d = zeros(N1, N2);

    for k = 1: Npeak
        tmp_t2 = exp((1i*Omg2(k)-R2(k))*t2);
        tmp_t1 = exp((1i*Omg1(k)-R1(k))*t1(:));
        tmp = tmp_t1*tmp_t2;
        fid_2d = fid_2d+A(k)*tmp;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% apply window %%%%
    w1 = nmrpipe_sp_window(N1,0,0.95,1,0,0,0.5)';
    w2 = nmrpipe_sp_window(N2,0,0.95,1,0,0,0.5);
    window = w1 * w2;
    if win_flag ==1 
        fid_2d = fid_2d .* window;
    end

    fid_2d = fid_2d/max(abs(fid_2d(:)));

    signal_power = mean(abs(fid_2d(:).^2));
    SNR_linear = 10^(SNR_dB/10);
    noise_power = signal_power /SNR_linear;
    noise_std = sqrt(noise_power/2);
    noise = noise_std*((randn(size(fid_2d)))+1j*randn(size(fid_2d)));
    nusr = nusr_min + round((nusr_max-nusr_min)*rand,2);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% add noise %%%%
    if noise_flag==1
        fid_2d_noise = fid_2d + noise;
    else
        fid_2d_noise = fid_2d;
    end

    mask = SNUSMask(N1,N2,nusr,'2Dpoisson'); % generate sampling mask nusr_min ~ nusr_max

    
    fid_2d_nus = fid_2d_noise.*mask;

    fid_2d_2c = zeros(N1,N2,2);
    fid_2d_nus_2c = zeros(N1,N2,2);
    f_2d_2c = zeros(N1,N2,2);
    f_2d_nus_2c = zeros(N1,N2,2);

    fid_2d_2c(:,:,1) = real(fid_2d);
    fid_2d_2c(:,:,2) = imag(fid_2d);
    fid_2d_nus_2c(:,:,1) = real(fid_2d_nus);
    fid_2d_nus_2c(:,:,2) = imag(fid_2d_nus);
    f_2d_2c(:,:,1) = real(fft2(fid_2d));
    f_2d_2c(:,:,2) = imag(fft2(fid_2d));
    f_2d_nus_2c(:,:,1) = real(fft2(fid_2d_nus));
    f_2d_nus_2c(:,:,2) = imag(fft2(fid_2d_nus));
    mask_2c = repmat(mask,[1,1,2]);
    mask_2c = double(mask_2c);

end


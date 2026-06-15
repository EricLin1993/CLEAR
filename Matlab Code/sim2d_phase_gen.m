function [fid_2d_2c,fid_2d_nus_2c,f_2d_2c,f_2d_nus_2c,mask_2c] = sim2d_phase_gen(A_min,A_max,N1,N2,nusr_min,nusr_max,SNR_dB_min,SNR_dB_max,noise_flag,win_flag,phase0_max,phase0_min)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % generate simulation data with phase distortion
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Npeak_min = 5;
    Npeak_max = 15;

    SW2_min = 1500;
    SW2_max = 3000;
    SW1_min = 1800;
    SW1_max = 5400;

    R1_min = 5;
    R1_max = 60;
    R2_min = 5;
    R2_max = 60;

    % generate one signal
    Npeak = Npeak_min + round(rand*(Npeak_max-Npeak_min));
    SNR_dB = SNR_dB_min + round(rand*(SNR_dB_max-SNR_dB_min));

    SW1 = SW1_min + round(rand*(SW1_max-SW1_min));
    SW2 = SW2_min + round(rand*(SW2_max-SW2_min));

    R1 = R1_min + rand(1, Npeak)*(R1_max-R1_min);
    R2 = R2_min + rand(1, Npeak)*(R2_max-R2_min);
    A  = A_min + rand(1, Npeak)*(A_max-A_min);
    
    Omg1 = rand(1, Npeak)*SW1*2*pi;  
    Omg2 = rand(1, Npeak)*SW2*2*pi;

    t1 = (0:N1-1)/SW1;
    t2 = (0:N2-1)/SW2;
    
    fid_2d = zeros(N1, N2);


    % ===== PH0 (zero-order phase)=====
    ph0_deg = phase0_min + rand*(phase0_max - phase0_min);
    PH0 = deg2rad(ph0_deg);
    

    % ===================== generate FID =====================
    for k = 1:Npeak
    
        tmp_t2 = exp((1i*Omg2(k) - R2(k)) * t2);
        tmp_t1 = exp((1i*Omg1(k) - R1(k)) * t1(:));
    
        tmp = tmp_t1 * tmp_t2;
    
        fid_2d = fid_2d + A(k) * tmp;
    
    end
    
    % ===== PH0 (global)=====
    fid_2d = fid_2d .* exp(1i * PH0);
    
    % ===================== apply window =====================
    w1 = nmrpipe_sp_window(N1,0,0.95,1,0,0,0.5)';
    w2 = nmrpipe_sp_window(N2,0,0.95,1,0,0,0.5);
    window = w1 * w2;
    
    if win_flag == 1
        fid_2d = fid_2d .* window;
    end
    
    % ===================== normalize =====================
    fid_2d = fid_2d / max(abs(fid_2d(:)));
    
    % ===================== noise =====================
    signal_power = mean(abs(fid_2d(:)).^2);
    SNR_linear = 10^(SNR_dB/10);
    noise_power = signal_power / SNR_linear;
    noise_std = sqrt(noise_power/2);
    
    noise = noise_std * (randn(size(fid_2d)) + 1i*randn(size(fid_2d)));
    
    if noise_flag == 1
        fid_2d_noise = fid_2d + noise;
    else
        fid_2d_noise = fid_2d;
    end
    
    % ===================== sampling mask =====================
    nusr = nusr_min + round((nusr_max - nusr_min)*rand,2);
    mask = SNUSMask(N1, N2, nusr, '2Dpoisson');
    
    fid_2d_nus = fid_2d_noise .* mask;
    
    % ===================== outputs =====================
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
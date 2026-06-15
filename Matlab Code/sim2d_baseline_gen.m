function [fid_2d_2c,fid_2d_nus_2c,f_2d_2c,f_2d_nus_2c,mask_2c] = sim2d_baseline_gen(A_min,A_max,N1,N2,nusr_min,nusr_max,SNR_dB_min,SNR_dB_max,noise_flag,win_flag,baseline_ratio,baseline_type)  % N1,N2,nusr_min,nusr_max,SNR_dB,noise_flag
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     generate simulation data with baseline distortion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     nusr_min      : Max Sampling Rate
%     nusr_max      : Min Sampling Rate
%     SNR_dB_min    : signal_noise_ratio  10   
%     SNR_dB_max    : 15;
%     noise_flag    : 1:noisy 0: clean
%     win_flag      :1:with SinBell  0: without window

    Npeak_min = 5; 
    Npeak_max = 15;  

    SW2_min = 1500; 
    SW2_max = 3000;
    SW1_min = 1800;
    SW1_max = 5400;

    R1_min = 5;
    R1_max = 10;
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
    Omg1 = rand(1, Npeak)*SW1*2*pi;  
    Omg2 = rand(1, Npeak)*SW2*2*pi;

    
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
    %%%% 选择是否加窗 %%%%
    w1 = nmrpipe_sp_window(N1,0,0.95,1,0,0,0.5)';
    w2 = nmrpipe_sp_window(N2,0,0.95,1,0,0,0.5);
    window = w1 * w2;
    if win_flag ==1 % && rand < 0.5 
        fid_2d = fid_2d .* window;
    end

    fid_2d = fid_2d/max(abs(fid_2d(:)));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Add Frequency Domain Baseline
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if baseline_ratio > 0
        spec_2d = fftshift(fft2(fid_2d));
        max_spec = max(abs(spec_2d(:))); 
    
        x = linspace(-1, 1, N2);
        
        % different baseline types
        switch lower(baseline_type)
                
            case 'quadratic'   
                baseline_1d = x.^2 - mean(x.^2);

            case 'sine'        % Baseline Roll
                f = 1 + 2 * rand();   
                phi = rand() * 2 * pi; 
                baseline_1d = sin(2 * pi * f * x + phi);
                
            otherwise
                baseline_1d = zeros(size(x));
        end
        
        random_phase = exp(1i * 2 * pi * rand()); 
        baseline_2d = repmat(baseline_1d * random_phase, N1, 1);

        baseline_2d = baseline_2d / max(abs(baseline_2d(:))); 
        baseline_2d = baseline_2d * max_spec * baseline_ratio; 
    
        spec_2d = spec_2d + baseline_2d;
        fid_2d_baseline = ifft2(ifftshift(spec_2d));
        
        fid_2d_baseline = fid_2d_baseline / max(abs(fid_2d(:)));
    end
    

    signal_power = mean(abs(fid_2d(:).^2));
    SNR_linear = 10^(SNR_dB/10);
    noise_power = signal_power /SNR_linear;
    noise_std = sqrt(noise_power/2);
    noise = noise_std*((randn(size(fid_2d)))+1j*randn(size(fid_2d)));
    nusr = nusr_min + round((nusr_max-nusr_min)*rand,2);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Add Noise %%%%
    if noise_flag==1
        fid_2d_noise = fid_2d_baseline + noise;
    else
        fid_2d_noise = fid_2d_baseline;
    end
    

    mask = SNUSMask(N1,N2,nusr,'2Dpoisson'); 

    
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


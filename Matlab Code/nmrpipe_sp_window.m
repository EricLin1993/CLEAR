function w = nmrpipe_sp_window(N, off, en, pow, elb, glb, c)
    % N   : Number of FID points
    % off : Window start fraction (0~1)
    % en  : Window end fraction (0~1)
    % pow : Power of the sine window
    % elb : Early line broadening (Hz or relative units)
    % glb : Global line broadening
    % c   : Multiply the first point of the FID by 0.5

    
    x = 0:1:N-1;
    w = sin(pi*off + pi*(en-off)/(N-1)*x).^pow;
    if elb ~= 0 || glb ~= 0
        w = w .* exp(-elb * x - glb * x);
    end
end

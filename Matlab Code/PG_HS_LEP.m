function [outputArg] = PG_HS_LEP(inputArg)
%  This matlab function, we named it PG_HS_LEP(Poisson Gap by Hyberts Sven and Lin Enping), 
%  is for generating the 1D,2D or 3D Poisson Gap(PG) schedules which were proposed by Hybert et al. in the fellowing original literatures: 
%  1. Hyberts, S. G.;  Takeuchi, K.; Wagner, G., Poisson-Gap Sampling and Forward Maximum Entropy Reconstruction for Enhancing the Resolution and Sensitivity of Protein NMR Data. Journal of the American Chemical Society 2010, 132 (7), 2145-2147.
%  2. Hyberts, S. G.;  Arthanari, H.; Wagner, G., Applications of Non-Uniform Sampling and Processing. In Novel Sampling Approaches in Higher Dimensional NMR, Billeter, M.; Orekhov, V., Eds. Springer Berlin Heidelberg: Berlin, Heidelberg, 2012; pp 125-148.
%  The C source code was provided by Dr.Sven Hyberts from Department of Biological Chemistry and Molecular Pharmacology, HarVard Medical School. 
%  Enping Lin used the mixing programming skill to make it work in the matlab platform.

%
%  Enping Lin    
%  email: enpinglin@qq.com
%  2023.1.5
%---------------------------------------------------------------------------------------------------------
   
    dim = inputArg.dim;
    n1=1;n2=1;n3=1;
    sn_tolerance = 0.01;
    sineportion = 1;
    if isfield(inputArg,'sineportion')
       sineportion =  inputArg.sineportion ;
    end  
    if isfield(inputArg,'n1')
       n1 = inputArg.n1;
    end    
    if isfield(inputArg,'n2')
       n2 = inputArg.n2;
    end    
    if isfield(inputArg,'n3')
       n3 = inputArg.n3;
    end    
    if isfield(inputArg,'sn_tolerance')
       sn_tolerance = inputArg.sn_tolerance;
    end 
    if inputArg.sr<=1 && inputArg.sr >= 0
      sr = inputArg.sr;
      sn = round(n1*n2*n3*sr);
    else 
       error("The sampling rate should be within [0,1]");
    end
    if dim == 1 || dim == 2
      outputArg = PG_revised(dim,0,sineportion ,sn,sn_tolerance,n1,n2,n3,0);  %%%  PG_revised15
      outputArg = logical(outputArg);
      fprintf("The dim is %d and the output is a binary mask\n", dim );
    elseif   dim == 3
      outputArg = PG_revised(dim,0,2,sn,sn_tolerance,n1,n2,n3,0);  %%%
      fprintf("The dim is %d and the output is an index matrix\n The index triplets are collumn-wise\n", dim );
    else
        error("The dim could only be 1, 2 or 3. ");
    end
end


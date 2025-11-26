function [ mask ] = SNUSMask( N1,N2,nusr,smodel )


 switch smodel
     
     case '2Dpoisson'   
%          addpath('PoissonRandNum_2')
%          [mask, n] = PoissonRandNum(2, 0, 1, round(nusr*N*N), 1e-5, N, N, 0, 1);
%
%         [ mask ] = Poisson2D( N,N,nusr );
% 
%           load('Mask2dPoisson.mat');
%           mask = MASK{1};

%         rtn = randi(100);
%         rr = textread(['\home\eeeefei\Desktop\',num2str(rtn),'.txt']);
%         mask = zeros(32);
%         for it =1:size(rr,1)
%            mask(rr(it,1)+1,rr(it,2)+1) = 1;  
%         end    
         inputArg.dim=2;
         inputArg.n1=N1;
         inputArg.n2=N2;
         inputArg.n3=1;
         inputArg.sr=nusr;
         inputArg.sineportion = 1;
     
         [ mask] = PG_HS_LEP(inputArg);

     case 'sym'   
        c = N;
        n = c*(c+1)/2;
        m = zeros(n,1); 
        l = round(nusr*N*N);
        ind = randperm(n,l);
        m(ind)=1;
        m = logical(m);
        upf = zeros(n,1);
        upf(rand(size(upf))<0.5)=1;
        [ mask ] = SymMask( m,upf,c);
     case 'ran'
         l = round(nusr*N*N);
         mask = zeros(N,N);
         ind = randperm(N*N,l);
         mask(ind)=1;
     case 'sympoisson'    
        c = N1;
        n = c*(c+1)/2;
        
        m = zeros(n,1);
        [v,k] = Poisson1D(1,round(N1*N1*nusr),n);
        m(v)=1;
        m = logical(m);
        
                
%         [ mt ] = Poisson2D( N,N,2*nusr );
%         flag = logical(triu(ones(N),0));
%         m = mt(flag);
%         
        
        upf = zeros(n,1);
        upf(rand(size(upf))<0.5)=1;
        [ mask ] = SymMask( m,upf,c);

     otherwise  
        disp('smodel is set ran in default...')
         n = N1*N2;
         l = round(nusr*n);
         mask = zeros(N1,N2);
         ind = randperm(n,l);
         mask(ind)=1;
 end

end


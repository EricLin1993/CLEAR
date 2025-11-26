function [ X,Xdiff,OV,RecTime ] = NUS2D_MF( Y,Mask,lambda,Wyes )
%----------------------------------------------------------------------
%----------------------------------------------------------------------
% NUS2D_MF to solve
%                     arg(X) min 1/2 * (||Mask.*(X-Y)||_2).^2 + lambda/2*(||A||F2+||B||F2)
%                         s.t. HX = A*B    
%                      where, H is the operator of block Hankel matrix
%                      pertutation, 1/2*(||A||F2+||B||F2) = ||HX||_*. 
% Input :
%        Y : 2D Undersampled data;
%          Mask : sampled mask;
%          lambda : regularized parameter
%   Output :
%          X: reconstructed data;
%          Xdiff: the differentiation between neighbor iteration X;
%          OV: Objective function value through iteration;
% Authored by Enping Lin
% log:
% Date 
%      2022.2.3
%----------------------------------------------------------------------
%----------------------------------------------------------------------

    tic;
    maxloop = 100;
    sr = ceil(size(Y,2)/2);
    sc = size(Y,2) - sr+1;
    n1 = ceil(size(Y,1)/2);
    n2 = sr;
     %   Generate HhH weight   
    n3 = size(Y,1) - n1+1;
    t1 = min(n1,n3)-1;
    t2 = max(n1,n3)-t1;
    rw = ([1:1:t1, (t1+1)*ones(1,t2),t1:-1:1]).';
    
    t1 = min(sr,sc)-1;
    t2 = max(sr,sc)-t1;
    cw = ([1:1:t1, (t1+1)*ones(1,t2),t1:-1:1]);
    HhH = rw*cw;
  % ---------------------------------------


    Temp = Matrix2BHankel(ones(size(Mask)),n1,n2);
    HhH = BHankel2Matrix(Temp,sr,sc );
    if Wyes
        WMask = HhH .* Mask;
    else
        WMask = Mask;
    end
    mu = 1;

    % ---------------- Initialization --------------------
%     Yz = Matrix2BHankel(Y,n1,n2);
%     [U,S,V]=svd(Yz,'econ');
%     vs = diag(S);
%     vs = vs./max(vs); 
%     d = 20;
% %     d = round(length(vs)*1);
%     A=U(:,1:d)*(S(1:d,1:d).^(1/2));
%     B=(S(1:d,1:d).^(1/2))*V(:,1:d)';

    st= 20;
    A=rand(sr*n1,st);
    B=rand(st,sc*n3);

    D = zeros(sr*n1,sc*n3);
   
    for itloop = 1:maxloop 
        fprintf('Iteration: %d\n',itloop)
        % ------------ Update X ------------------
        Xup = WMask.*Y + BHankel2Matrix( mu*A*B-D,sr,sc );
        Xdown = 1*WMask+mu*HhH;
        X = Xup./Xdown;
        if itloop >= 2
           Xdiff(itloop-1) = norm(X-Xlast,'fro')/norm(X,'fro') ;
        end 
        HX = Matrix2BHankel(X,n1,n2);
        Xlast = X;
        
        % ------------ Update A,B ------------------
         % A new
         Bh = B';
         BBh = B*B';
         A = (mu*HX+D)*Bh*inv(lambda*eye(size(BBh))+mu*BBh);

        % B new
         Ah=A';
         AhA = Ah*A;
  
        B = inv(lambda*eye(size(AhA))+mu*AhA) * (Ah*(D+mu*HX));%mu/(alpha*lambda)*
        


        % ------------ Update D ------------------
        D = D+mu*(HX-A*B);

    
        
        % ------------ Estimate Objective Value --------------
        
        Nu = 1/2*( norm(B,'fro')+norm(A,'fro') );
        OV(itloop) = 0.5*( norm(Mask.*(X-Y),'fro') ).^2 + lambda*Nu;
    end
    RecTime = toc/60;
    fprintf('Finish Iteration: %d, Time lapse:%5.2f min \n',itloop,RecTime)
end


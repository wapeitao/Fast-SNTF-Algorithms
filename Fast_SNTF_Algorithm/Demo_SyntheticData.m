
clc; clear; close all

I = 50;
J = 10;
R = J;
SNR = 10; 

maxIter = 1e10;
maxTime = 500;
tol = 1e-5;
computeobj = false;
 
% Monte Carlo tests
MCT = 1; NumofMCT = 20; 
while MCT <= NumofMCT

    % generating the syntheic data
    Gtrue = rand(I,J);
    Ytensor = ktensor({Gtrue,Gtrue,Gtrue}); 
    Y = double(tensor(Ytensor));

    N = zeros(I,I,I);
    for i = 1:I
        for j = i:I
            for k = j:I
                randnum = randn(1);
                N(i,j,k) = randnum;
                N(i,k,j) = randnum;
                N(j,i,k) = randnum;
                N(j,k,i) = randnum;
                N(k,i,j) = randnum;
                N(k,j,i) = randnum;
            end
        end
    end
    sN = norm(N(:));
    sY = norm(Y(:));
    ratio = sY/(sN*sqrt( 10^(SNR/10) ));
    Y = max(Y+N*ratio, 0);

    % initialization
    G0 = rand(I,R)+1E-5;
    
    % performing the proposed multiplicative algorithms
    [G5,f5,t5,fit5] = Parallel_Multi_SNTF(Y,G0,(1/5),maxIter,maxTime,tol,computeobj);
    [G_alpha,f_alpha,t_alpha,fit_alpha] = alpha_Paralle_Multi_SNTF(Y,G0,(1/2),maxIter,maxTime,tol); 
    [G_beta,f_beta,t_beta,fit_beta] = beta_Paralle_Multi_SNTF(Y,G0,(1/2),maxIter,maxTime,tol);
   
    T5(MCT,1) = t5(end);
    T_alpha(MCT,1) = t_alpha(end);
    T_beta(MCT,1) = t_beta(end);   

    Fit5(MCT,1) = fit5(end);
    Fit_alpha(MCT,1) = fit_alpha(end);
    Fit_beta(MCT,1) = fit_beta(end);

    MCT = MCT+1;

end
        
disp('The results are shown as follows:')
meanT = [mean(T5),mean(T_alpha),mean(T_beta)]
stdT = [std(T5),std(T_alpha),std(T_beta)]

meanFIT = [mean(Fit5),mean(Fit_alpha),mean(Fit_beta)]
stdFIT = [std(Fit5),std(Fit_alpha),std(Fit_beta)]

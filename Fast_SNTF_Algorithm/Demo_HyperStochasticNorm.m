% Comparison between two hyper-stochastic normalization schemes

clc; clear; close all 

dataset = 'COIL';
K = 4;

switch dataset
    case {'COIL'}
          load COIL20_32x32.mat 
    case {'ORL'}
          load ORL_32x32.mat
    case {'UMIST'}
          load umist.mat 
          fea = X';
    case {'YALE'}
          load Yale_32x32.mat
    otherwise        
          error('the dataset does not exist!');
end    

fulldata = fea';
numperclass = 10;
maxIter = 1e10;
maxTime = 50;
tol = 1e-5;
computeobj = false;
tau = 0.1;

%% Monte Carlo tests
MCT = 1; NumofMCT = 20; 
while MCT <= NumofMCT
    
    % generate data subset
    [datasub,subN,labelsub,Trans_datasub,clist]=loadsub(fulldata,gnd,K,numperclass); 
       
    X = datasub;  
    label = labelsub;
          
    [U,~,latent] = pca(X');
    X = U(:,1:K)'*X;
    
    %% construct affinity tensor 

    [nfea,ndata] = size(X);
    
    X2 = sum(X.^2);
    D = repmat(X2,ndata,1)+repmat(X2',1,ndata)-2*X'*X;
    D = real(sqrt(D));
  
    % construct 3-way affinity tensor
    V = zeros([ndata,ndata,ndata]);
    for i = 1:ndata
        for j = i:ndata  
            for k = j:ndata 
                temp = D(i,j)+D(i,k)+D(j,k);
                V(i,j,k) = temp;
                V(i,k,j) = temp;
                V(j,i,k) = temp;
                V(j,k,i) = temp;
                V(k,i,j) = temp;
                V(k,j,i) = temp;
            end
        end
    end
    
    V2 = V.^2;
    delta = sort(V2(:),1,'ascend');
    delta = delta(floor(tau*(ndata^3)))/3;
    Y = exp(-1*V2/delta);
    
    % [1] using the hyper-stochastic normalization scheme of Shashua et al.
    Y1 = max(HyperStochasticTensor(Y),1E-12);
    
    % [2] using the proposed hyper-stochastic normalization scheme
    Y2 = max(nearestHS(Y),1E-12);
          
    %% perform SNTF 
    
    for init_iter = 1:5
        
        disp([MCT,init_iter])
        
        G0 = rand(ndata,K)+1e-5; 
                        
        % [1]
        [G5_Y1,f5_Y1,t5_Y1,fit5_Y1] = Parallel_Multi_SNTF(Y1,G0,(1/5),maxIter,maxTime,tol,computeobj);
        [G_alpha_Y1,f_alpha_Y1,t_alpha_Y1,fit_alpha_Y1] = alpha_Paralle_Multi_SNTF(Y1,G0,(1/2),maxIter,maxTime,tol);
        [G_beta_Y1,f_beta_Y1,t_beta_Y1,fit_beta_Y1] = beta_Paralle_Multi_SNTF(Y1,G0,(1/2),maxIter,maxTime,tol); 
        
        [AC_SNTF5_Y1(init_iter,MCT),MIhat_SNTF5_Y1(init_iter,MCT)] = AC_MIhat(G5_Y1,label,K);
        [AC_SNTF_alpha_Y1(init_iter,MCT),MIhat_SNTF_alpha_Y1(init_iter,MCT)] = AC_MIhat(G_alpha_Y1,label,K);
        [AC_SNTF_beta_Y1(init_iter,MCT),MIhat_SNTF_beta_Y1(init_iter,MCT)] = AC_MIhat(G_beta_Y1,label,K);
              
        T_SNTF5_Y1(init_iter,MCT) = t5_Y1(end);
        T_SNTF_alpha_Y1(init_iter,MCT) = t_alpha_Y1(end);
        T_SNTF_beta_Y1(init_iter,MCT) = t_beta_Y1(end);   
         
        % [2]
        [G5_Y2,f5_Y2,t5_Y2,fit5_Y2] = Parallel_Multi_SNTF(Y2,G0,(1/5),maxIter,maxTime,tol,computeobj);
        [G_alpha_Y2,f_alpha_Y2,t_alpha_Y2,fit_alpha_Y2] = alpha_Paralle_Multi_SNTF(Y2,G0,(1/2),maxIter,maxTime,tol);
        [G_beta_Y2,f_beta_Y2,t_beta_Y2,fit_beta_Y2] = beta_Paralle_Multi_SNTF(Y2,G0,(1/2),maxIter,maxTime,tol); 
        
        [AC_SNTF5_Y2(init_iter,MCT),MIhat_SNTF5_Y2(init_iter,MCT)] = AC_MIhat(G5_Y2,label,K);
        [AC_SNTF_alpha_Y2(init_iter,MCT),MIhat_SNTF_alpha_Y2(init_iter,MCT)] = AC_MIhat(G_alpha_Y2,label,K);
        [AC_SNTF_beta_Y2(init_iter,MCT),MIhat_SNTF_beta_Y2(init_iter,MCT)] = AC_MIhat(G_beta_Y2,label,K);
              
        T_SNTF5_Y2(init_iter,MCT) = t5_Y2(end);
        T_SNTF_alpha_Y2(init_iter,MCT) = t_alpha_Y2(end);
        T_SNTF_beta_Y2(init_iter,MCT) = t_beta_Y2(end);   
           
    end
           
    MCT = MCT+1;
    
end
% [1]
[bestAC_SNTF5_Y1, index_SNTF5_Y1] = max(AC_SNTF5_Y1);
bestMIhat_SNTF5_Y1 = MIhat_SNTF5_Y1((0:NumofMCT-1)*5+index_SNTF5_Y1);
bestT_SNTF5_Y1 = T_SNTF5_Y1((0:NumofMCT-1)*5+index_SNTF5_Y1);

[bestAC_SNTF_alpha_Y1, index_SNTF_alpha_Y1] = max(AC_SNTF_alpha_Y1);
bestMIhat_SNTF_alpha_Y1 = MIhat_SNTF_alpha_Y1((0:NumofMCT-1)*5+index_SNTF_alpha_Y1);
bestT_SNTF_alpha_Y1 = T_SNTF_alpha_Y1((0:NumofMCT-1)*5+index_SNTF_alpha_Y1);

[bestAC_SNTF_beta_Y1, index_SNTF_beta_Y1] = max(AC_SNTF_beta_Y1);
bestMIhat_SNTF_beta_Y1 = MIhat_SNTF_beta_Y1((0:NumofMCT-1)*5+index_SNTF_beta_Y1);
bestT_SNTF_beta_Y1 = T_SNTF_beta_Y1((0:NumofMCT-1)*5+index_SNTF_beta_Y1);

% [2]
[bestAC_SNTF5_Y2, index_SNTF5_Y2] = max(AC_SNTF5_Y2);
bestMIhat_SNTF5_Y2 = MIhat_SNTF5_Y2((0:NumofMCT-1)*5+index_SNTF5_Y2);
bestT_SNTF5_Y2 = T_SNTF5_Y2((0:NumofMCT-1)*5+index_SNTF5_Y2);

[bestAC_SNTF_alpha_Y2, index_SNTF_alpha_Y2] = max(AC_SNTF_alpha_Y2);
bestMIhat_SNTF_alpha_Y2 = MIhat_SNTF_alpha_Y2((0:NumofMCT-1)*5+index_SNTF_alpha_Y2);
bestT_SNTF_alpha_Y2 = T_SNTF_alpha_Y2((0:NumofMCT-1)*5+index_SNTF_alpha_Y2);

[bestAC_SNTF_beta_Y2, index_SNTF_beta_Y2] = max(AC_SNTF_beta_Y2);
bestMIhat_SNTF_beta_Y2 = MIhat_SNTF_beta_Y2((0:NumofMCT-1)*5+index_SNTF_beta_Y2);
bestT_SNTF_beta_Y2 = T_SNTF_beta_Y2((0:NumofMCT-1)*5+index_SNTF_beta_Y2);

[mean(bestAC_SNTF5_Y1),mean(bestAC_SNTF_alpha_Y1),mean(bestAC_SNTF_beta_Y1);
 mean(bestAC_SNTF5_Y2),mean(bestAC_SNTF_alpha_Y2),mean(bestAC_SNTF_beta_Y2)]
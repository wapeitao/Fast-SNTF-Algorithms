clc; clear; close all

nfea = 2;
ndata = 80;

maxIter = 1e10;
maxTime = 500;
tol = 1e-5;
computeobj = false;

MCT = 1;  NumofMCT = 20; 
while MCT <= NumofMCT
      
    % generate data      
    a = randn(nfea,4); 
    b = randn(nfea,4); 
    z = 2*rand(1,ndata/4)-1;
    x1 = a(:,1)*z + repmat(b(:,1),1,ndata/4);
    z = 2*rand(1,ndata/4)-1;
    x2 = a(:,2)*z + repmat(b(:,2),1,ndata/4);
    z = 2*rand(1,ndata/4)-1;
    x3 = a(:,3)*z + repmat(b(:,3),1,ndata/4);
    z = 2*rand(1,ndata/4)-1;
    x4 = a(:,4)*z + repmat(b(:,4),1,ndata/4);
    X = [x1,x2,x3,x4];
    label = [ones(1,ndata/4),2*ones(1,ndata/4),3*ones(1,ndata/4),4*ones(1,ndata/4)]';
        
    %% construct affinity tensor 

    K = max(label);
    
    % construct 3-way affinity tensor
    tau = 0.1;
    V = zeros([ndata,ndata,ndata]);
    for i = 1:ndata
        for j = i+1:ndata  
            for k = j+1:ndata  
                temp = similarity(X,[i,j,k]);
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
    temp = sort(V2(:),1,'ascend');
    delta = temp(floor(tau*ndata^3))/3;
    Y = exp(-1*V2/delta);
    Y = nearestHS(Y);
    Y = max(Y,1E-12);
        
    
    %% performing the proposed multiplicative algorithms
    
    for init_iter = 1:5
        
        disp([MCT,init_iter])
        
        % initialization
        G0 = rand(ndata,K)+1e-5; 
                                  
        [G5,f5,t5,fit5] = Parallel_Multi_SNTF(Y,G0,(1/5),maxIter,maxTime,tol,computeobj);       
        [G_alpha,f_alpha,t_alpha,fit_alpha] = alpha_Paralle_Multi_SNTF(Y,G0,(1/2),maxIter,maxTime,tol);
        [G_beta,f_beta,t_beta,fit_beta] = beta_Paralle_Multi_SNTF(Y,G0,(1/2),maxIter,maxTime,tol);  
        
        [AC_SNTF5(init_iter,MCT),MIhat_SNTF5(init_iter,MCT)] = AC_MIhat(G5,label,K);
        [AC_SNTF_alpha(init_iter,MCT),MIhat_SNTF_alpha(init_iter,MCT)] = AC_MIhat(G_alpha,label,K);
        [AC_SNTF_beta(init_iter,MCT),MIhat_SNTF_beta(init_iter,MCT)] = AC_MIhat(G_beta,label,K);
   
        T_SNTF5(init_iter,MCT) = t5(end);
        T_SNTF_alpha(init_iter,MCT) = t_alpha(end);
        T_SNTF_beta(init_iter,MCT) = t_beta(end);  
                                     
    end
                       
    MCT = MCT+1;
    
end

[bestAC_SNTF5, index_SNTF5] = max(AC_SNTF5);
bestMIhat_SNTF5 = MIhat_SNTF5((0:NumofMCT-1)*5+index_SNTF5);
bestT_SNTF5 = T_SNTF5((0:NumofMCT-1)*5+index_SNTF5);

[bestAC_SNTF_alpha, index_SNTF_alpha] = max(AC_SNTF_alpha);
bestMIhat_SNTF_alpha = MIhat_SNTF_alpha((0:NumofMCT-1)*5+index_SNTF_alpha);
bestT_SNTF_alpha = T_SNTF_alpha((0:NumofMCT-1)*5+index_SNTF_alpha);

[bestAC_SNTF_beta, index_SNTF_beta] = max(AC_SNTF_beta);
bestMIhat_SNTF_beta = MIhat_SNTF_beta((0:NumofMCT-1)*5+index_SNTF_beta);
bestT_SNTF_beta = T_SNTF_beta((0:NumofMCT-1)*5+index_SNTF_beta);

%% show the results

AC_SNTF = [bestAC_SNTF5',bestAC_SNTF_alpha',bestAC_SNTF_beta'];
[mean(AC_SNTF);std(AC_SNTF)]

MIhat_SNTF = [bestMIhat_SNTF5',bestMIhat_SNTF_alpha',bestMIhat_SNTF_beta'];
[mean(MIhat_SNTF);std(MIhat_SNTF)]

T_SNTF = [bestT_SNTF5',bestT_SNTF_alpha',bestT_SNTF_beta'];
[mean(T_SNTF);std(T_SNTF)]

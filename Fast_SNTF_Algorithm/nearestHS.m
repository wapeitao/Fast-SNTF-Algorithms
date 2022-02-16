function F = nearestHS(A, aSum, precision, maxLoops)
% Find the nearest hyper-stochastic tensor to A in least squares error norm

if (nargin < 4)
    maxLoops = 1000;
    if (nargin < 3)
        precision = 0.001;
        if (nargin < 2)
            aSum = 1;
        end
    end
end

ndim = size(A,1);
Amat = reshape(A,[ndim,ndim^2]);
one_vec = ones(ndim,1);
coef1 = 1/(ndim^2); 
coef2 = 2/(3*ndim^3);   

F = A;
F = F * (aSum / mean(sum(Amat,2)));

for i = 1 : maxLoops
        
    Fmat = reshape(F,[ndim,ndim^2]);
    temp = sum(Fmat,2);    
    mu = coef1*aSum-coef2*ndim*aSum-coef1*temp+coef2*sum(temp);
        
    S = ktensor({mu,one_vec,one_vec})+ktensor({one_vec,mu,one_vec})+ktensor({one_vec,one_vec,mu}); 
    S = double(tensor(S));
    
	F = F+S;
               
	F(find(F<0)) = 0;
     
    if ( max(abs( sum(sum(F))-aSum )) < precision )
        break;
    end
        
end




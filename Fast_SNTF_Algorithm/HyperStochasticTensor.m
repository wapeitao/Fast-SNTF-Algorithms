function F = HyperStochasticTensor(A, precision)
% Convert the third-order tensor A to a hyper-stochastic tensor by the scheme in 
% "Multiway Clustering Using Supersymmetric Nonnegative Tensor Factorization"

if (nargin < 2)
    precision = 0.001;
end

N = size(A,1);
F = A; 
Fmat = reshape(F,[N,N^2]);
while true

    a = sum(Fmat,2);
    a = a.^(1/3);
    D = diag(1./a);
    
    Fmat = D*Fmat;
    Fmat = reshape(Fmat',[N,N^2]);
    Fmat = D*Fmat;
    Fmat = reshape(Fmat',[N,N^2]);
    Fmat = D*Fmat;
    Fmat = reshape(Fmat',[N,N^2]);
    
    if ( max(abs( sum(Fmat,2)-1 )) < precision )
        break;
    end

end
F = reshape(Fmat,[N,N,N]);

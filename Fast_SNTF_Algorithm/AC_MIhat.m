function [AC,MIhat] = AC_MIhat(W,gnd,nClass)

[~,ind]=max(W,[],2);
% ind = litekmeans(W,nClass,'Replicates',20);
res = bestMap(gnd,ind);
AC = length(find(gnd == res))/length(gnd);
MIhat = MutualInfo(gnd,res);
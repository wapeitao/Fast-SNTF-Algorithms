function [datasub,subN,labelsub,Trans_datasub,clist]=loadsub(fulldata,gnd,k,numperclass)

Class = unique(gnd);
turelabel = gnd;
p = randperm(length(Class));
clist = Class(p(1:k));

sublist=[]; labelsub=[];
for i=1:k
    temp=find(turelabel==clist(i));
    if numperclass < length(temp)
       p1 = randperm(length(temp));
       temp = temp(p1(1:numperclass));
    end
    sublist=[sublist;temp];
    labelsub=[labelsub;i*ones(length(temp),1)];
end

datasub=fulldata(:,sublist);
Trans_datasub=datasub';   
subN=length(labelsub);



function sim = similarity(X,C)

% calculate a line fit the three point
FO = fit(X(1,C)',X(2,C)','poly1');

%get parameter of fitting line
a = FO.p1;
b = FO.p2;

x1 = X(1,C(1)); 
x2 = X(1,C(2));
x3 = X(1,C(3));
y1 = X(2,C(1)); 
y2 = X(2,C(2));
y3 = X(2,C(3));
%formulation: y = a*x + b;
if (~isinf(a))
   dist = ((abs(y1-a*x1-b))+(abs(y2-a*x2-b))+ ...
           (abs(y3-a*x3-b)))/sqrt(a*a+1);
   sim = dist/3;
else
   sim = 0.5;    
end

end

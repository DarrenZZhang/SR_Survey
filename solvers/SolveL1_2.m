function [id x it] = SolveL1_2(A,b,par,train_num,L)
% Exploiting the FISTA to solve L_(1/2) norm minimization problem
x = zeros(size(A,2),1);
y = x;
f = norm(A*y - b,2)^2/2 + par.lambda*sum(sqrt(abs(y)));
t = 1;
tau = .95/L;
for it = 1:2000
    x_old = x;
    t_old = t;
    f_old = f;
    x = y - tau*A'*(A*y - b);
    phi = acos((abs(x)/3).^(-3/2)*par.lambda/8);    
    x = (1 + cos((pi-phi)*2/3)).*x*2/3;
    t = (1 + sqrt(1 + 4*t_old^2)) / 2; 
    y = x + (t_old - 1)/t * (x - x_old);
    f = norm(A*y - b,2)^2/2 + par.lambda*sum(sqrt(abs(y)));
    if abs(f - f_old)/abs(f_old) < 1e-4
        break
    end
end
x = reshape(x,train_num,par.class_num);
errors = zeros(1,par.class_num);
for ii = 1:par.class_num
    errors(ii) = norm(A(:,(ii-1)*train_num+1:ii*train_num)*x(:,ii)-b,2);
end
id = find(errors==min(errors));
x = x(:);
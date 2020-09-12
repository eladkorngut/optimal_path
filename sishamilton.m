function dydt=sishamilton(t,y)
b=5;
k=1;
dydt = [b*(1-y(1))*y(1)*exp(y(2))-k*y(1)*exp(-y(2)); b*(2*y(1)-1)*(exp(y(2))-1)-k*(exp(-y(2))-1)];
end
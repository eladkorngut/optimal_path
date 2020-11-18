function [Theory,D_analytical] = clancy_two_states(lam,mu,N)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
x0=(lam-1)/lam;
f=1/2;
% slope_approx=x0/lam-x0*((lam^2)*x0+2*lam*x0+2)/(2*(x0*lam+1)^2);
% D = (-mu(1)-mu(2)+lam*mu(1)*mu(2)+sqrt((mu(1)-mu(2))^2+(lam^2)*mu(1)*mu(2)))/(2*mu(1)*mu(2));
s0=(1/lam)+log(lam)-1;
syms d x l
for j=1:length(mu)
    delta=(mu(j,2)-mu(j,1))/2;
    D_simp(j)=lam/2+(sqrt(4*delta^2+(lam^2)*(1-delta^2)^2)-2)/(2*(1-delta^2));
    solvp=vpasolve(mu(j,1)/(1+mu(j,1)*d)+mu(j,2)/(1+mu(j,2)*d)==2/lam,d);
    D_numerical(j)=solvp(2);
    D_analytical(j)=(-mu(j,1)-mu(j,2)+lam*mu(j,1)*mu(j,2)+sqrt((mu(j,1)-mu(j,2))^2+(lam^2)*(mu(j,1)^2)*(mu(j,2)^2)))/(2*mu(j,1)*mu(j,2));
    A_analytical(j)=(1/2)*log(1+mu(j,1)*D_analytical(j))+(1/2)*log(1+mu(j,2)*D_analytical(j))-(1/lam)*D_analytical(j);
    A_numerical(j)=(1/2)*log(1+mu(j,1)*D_numerical(j))+(1/2)*log(1+mu(j,2)*D_numerical(j))-(1/lam)*D_numerical(j);
    A_simp(j)=(1/2)*log(1+mu(j,1)*D_simp(j))+(1/2)*log(1+mu(j,2)*D_simp(j))-(1/lam)*D_simp(j);
    Theory(j)=N*A_simp(j);
%     A_self_dev=(2*(1+D_analytical*mu(j,1))*(1+D_analytical*mu(j,2))*log(1/2)-(1+D_analytical*mu(j,2))*log(1/(2+2*D_analytical*mu(j,1)))+D_analytical*mu(j,1)*(1+D_analytical*mu(j,2))*(-1+log(lam*mu(j,1))-log((d*mu(j,1))/(2+2*D_analytical*mu(j,1))))-(1+D_analytical*mu(j,1))*log(1/(2+2*D_analytical*mu(j,2)))+D_analytical*(1+D_analytical*mu(j,1))*mu(j,2)*(-1+log(lam*mu(j,2))-log(D_analytical/(2+2*D_analytical*mu(j,2))))+D_analytical*(mu(j,1)+mu(j,2)+2*D_analytical*mu(j,1)*mu(j,2))*log((d/2)*(mu(j,1)/(1+D_analytical*mu(j,1))+mu(j,2)/(1+D_analytical*mu(j,2)))))/(2*(1+D_analytical*mu(j,1))*(1+D_analytical*mu(j,2)));
end


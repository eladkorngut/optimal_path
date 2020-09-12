function [] = find_pu_pw_homo_network()
%This function finds and plots the values of pu and pw in the homo degree
%hetro rate case
%   No arguments yt
epsilon=linspace(0.01,0.8,10);
Lambda=linspace(1.01,2.8,10);
l_for_eps=1.6;
eps_for_l=0.1;
pu_eps=zeros(size(epsilon));
pw_eps=zeros(size(epsilon));
for e=1:length(epsilon)
   [pu_eps(e),pw_eps(e)]=func_p_eq_point(epsilon(e),l_for_eps);
end

pu_lambda=zeros(size(Lambda));
pw_lambda=zeros(size(Lambda));

for l=1:length(Lambda)
    [pu_lambda(l),pw_lambda(l)]=func_p_eq_point(eps_for_l,Lambda(l));
end
subplot(1,2,1)
plot(epsilon,pu_eps,'LineWidth',10,'Color','r','LineStyle','-');
hold on
plot(epsilon,pw_eps,'LineWidth',10,'Color','b','LineStyle','-.');
xlabel('\epsilon')
ylabel('p')
title('p vs \epsilon with constant \Lambda')
legend("p_u", "p_w")
set(gca,'FontSize',40,'FontWeight','bold');
subplot(1,2,2)
plot(Lambda,pu_lambda,'LineWidth',10,'Color','r','LineStyle','-');
hold on
plot(Lambda,-2*log(Lambda),'LineWidth',10,'Color','g','LineStyle','-');
plot(Lambda,pw_lambda,'LineWidth',10,'Color','b','LineStyle','- -');
xlabel('\Lambda')
ylabel('p')
title('p vs \Lambda with constant \epsilon')
legend("p_u","Theroy p_w=-2 ln(\Lambda)" ,"p_w")
set(gca,'FontSize',40,'FontWeight','bold');
end


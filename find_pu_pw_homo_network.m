function [pu,pw,pu_theory,pw_theory] = find_pu_pw_homo_network()
%This function finds and plots the values of pu and pw in the homo degree
%hetro rate case

epsilon=linspace(0.01,0.8,10);
Lambda=linspace(1.01,2.8,10);


for e=1:length(epsilon)
    for l=1:length(Lambda)
        [pu(e,l),pw(e,l)]=func_p_eq_point(epsilon(e),Lambda(l));
        pu_theory(e,l)=0;
        pw_theory(e,l)=-2*log(Lambda(l));
    end
end
figure(1)
hold on
m=['+','o','*','.','x','s','d','^','v','<'];
for i=1:length(pu)
    plot(epsilon,pu(:,i),'LineWidth',10,'Marker',m(i),'MarkerSize',30,'LineStyle','none');
end
xlabel('\epsilon','FontWeight','bold','FontSize',60);
ylabel('p_u','FontWeight','bold','FontSize',60);
title('p_u vs \epsilon for 10 meauserments','FontWeight','bold','FontSize',50);
legend('\Lambda=1.01','\Lambda=1.209','\Lambda=1.408','\Lambda=1.607','\Lambda=1.806','\Lambda=2.01','\Lambda=2.203','\Lambda=2.402','\Lambda=2.601','\Lambda=2.8')
set(gca,'FontSize',40,'FontWeight','bold');

p_diff=pw-pw_theory;

figure(2)
hold on
m=['+','o','*','.','x','s','d','^','v','<'];
for i=1:length(pw)
    plot(Lambda,p_diff(i,:),'LineWidth',10,'Marker',m(i),'MarkerSize',30,'LineStyle','none');
end
xlabel('\Lambda','FontWeight','bold','FontSize',60);
ylabel('p_w','FontWeight','bold','FontSize',60);
title('p_w-2*ln(\Lambda) vs \Lambda for 10 meauserments','FontWeight','bold','FontSize',50);
legend('\epsilon=0.01','\epsilon=0.0978','\epsilon=0.1856','\epsilon=0.2733','\epsilon=0.3611','\epsilon=0.4899','\epsilon=0.5367','\epsilon=0.6244','\epsilon=0.7122','\epsilon=0.8')
set(gca,'FontSize',40,'FontWeight','bold');

end


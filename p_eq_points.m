problem=false;
u=0;
w=0;
epsilon = 0.2;
Lambda = 1.3;
syms pw pu
dpw_dt = -1*((1/2)*Lambda*(epsilon+1)*(u-w+1)*(exp((1/2)*(pw-pu)-1))-(1/2)*Lambda*w*(epsilon+1)*(exp((1/2)*(pw-pu))-1)+(1/2)*Lambda*(1-epsilon)*(1-u-w)*(exp((1/2)*(pu+pw))-1)-(1/2)*Lambda*w*(1-epsilon)*(exp((1/2)*(pu+pw))-1)+(1/2)*(exp((1/2)*(-pu-pw))-1)+(1/2)*(exp((1/2)*(pu-pw)-1)))==0;
dpu_dt = -1*((1/2)*Lambda*w*(epsilon+1)*(exp((pw-pu)/2)-1)-(1/2)*Lambda*w*(1-epsilon)*(exp((pu+pw)/2)-1)+(1/2)*(exp((-pu-pw)/2)-1)+(1/2)*(1-exp((pu-pw)/2)))==0;
sol = solve([dpu_dt,dpw_dt],[pu,pw]);
pusol = real(double(sol.pu));
pwsol = real(double(sol.pw));
for i=1:length(pusol)
    if pusol(1)~=pusol(i)
        problem=true;
    end
end
for i=1:length(pwsol)
    if pwsol(1)~=pwsol(i)
        problem=true;
    end
end


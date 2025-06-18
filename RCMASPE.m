function [RCME]=RCMASPE(x,m,tau,scale)
%
% Calculates the refined composite multiscale Amplitude-sensitive permutation entropy (RCMASPE) 
%%
for j=1:scale
    pdf=[];
    for jj=1:j
        xs = Multi(x(jj:end),j);% 复合多尺度过程composite multiscale process
        [PE, T_pdf]=ASPE(xs,m,tau);
        pdf=[pdf ; T_pdf];
    end
    % refined过程
    pdf=mean(pdf,1);
    pdf=pdf(pdf~=0);
    RCME(j)=-sum(pdf .* log(pdf));
    
%     normalized
    RCME(j)=RCME(j)/log(factorial(m));
end

function M_Data = Multi(Data,S)
%  generate the consecutive coarse-grained time series
%  Input:   Data: time series;
%           S: the scale factor
% Output:
%           M_Data: the coarse-grained time series at the scale factor S

L = length(Data);
J = fix(L/S);

for i=1:J
    M_Data(i) = mean(Data((i-1)*S+1:i*S));
end

% 
% %% 设置 熵 的一些参数
% m = 3; % 嵌入维数 embedding dimension;
% tau = 1; % 时间延迟time delay
% scale = 4; % 尺度scale for time shift
% for s = 1 : 5
%     for t = 1 : 100 
%         signal = data{s,1}(t,:);
%         % haspe{t,s} = HASPE_dwt(signal,m,tau,'db4',5);
%         % tsmapse{t,s} = TSMASPE(signal,m,tau,10);
%         % rcmaspe{t,s} = RCMASPE(signal,m,tau,10); 
%         aspe{t,s} = ASPE(signal,m,tau);
%         pe{t,s} = PE(signal,m,tau);
%     end
% end


% features{1,1} = haspe;
% features{2,1} = tsmapse;
% features{3,1} = rcmaspe;
features{4,1} = aspe;
features{5,1} = pe;
AA = [];
BB = [];
CC = [];
DD = [];
EE = [];
labels = [zeros(200,1);ones(300,1)];
for i = 1 : 5
    AA = vertcat(AA,features{1,1}(:,i));
    BB = vertcat(CC,features{2,1}(:,i));
    CC = vertcat(CC,features{3,1}(:,i));
    DD = vertcat(DD,features{4,1}(:,i));
    EE = vertcat(EE,features{5,1}(:,i));
end

features_st.haspe = cell2mat(AA);
features_st.tsmaspe = cell2mat(BB);
features_st.rcmaspe = cell2mat(CC);
features_st.aspe = cell2mat(DD);
features_st.pe = cell2mat(EE);
function tsmaspe_vector = TSMASPE(data, m, t, k_max)
% TSMASPE 计算时移多尺度振幅敏感排列熵
%
% 输入:
%   data   - 一维时间序列行向量或列向量
%   m      - 用于ASPE计算的嵌入维数 (e.g., 3)
%   t      - 用于ASPE计算的时间延迟 (e.g., 1)
%   k_max  - 最大的时间间隔(尺度)因子 (e.g., 10)
%
% 输出:
%   tsmaspe_vector - 一个包含各时间间隔下平均ASPE值的行向量

    tsmaspe_vector = zeros(1, k_max);

    % 外层循环：遍历每个时间间隔k
    for k = 1:k_max
        entropies_for_k = zeros(1, k);
        % 内层循环：为当前k创建k个时移子序列
        for j = 1:k
            % 创建时移子序列
            sub_sequence = data(j:k:end);
            
            % 确保子序列长度足够进行ASPE计算
            if length(sub_sequence) > m*t
                entropies_for_k(j) = ASPE(sub_sequence, m, t);
            else
                entropies_for_k(j) = 0; % 或者NaN，表示无法计算
            end
        end
        
        % 对当前k的所有时移子序列的熵值取平均
        tsmaspe_vector(k) = mean(entropies_for_k);
    end
end
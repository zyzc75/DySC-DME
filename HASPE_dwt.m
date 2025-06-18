function haspe_vector = HASPE_dwt(data, m, t, wavelet_name, level)
% HASPE_dwt 计算基于离散小波变换的层次振幅敏感排列熵
%
% 输入:
%   data         - 一维时间序列行向量或列向量
%   m            - 用于ASPE计算的嵌入维数 (e.g., 3)
%   t            - 用于ASPE计算的时间延迟 (e.g., 1)
%   wavelet_name - 小波基名称, 字符串类型 (e.g., 'db4', 'sym8')
%   level        - 小波分解的层数 (e.g., 4)
%
% 输出:
%   haspe_vector - 一个包含各频带ASPE值的行向量
%                  格式: [H(A_L), H(D_L), H(D_{L-1}), ..., H(D_1)]
%                  其中L是分解层数level

    % 1. 使用DWT进行多层分解
    [C, L] = wavedec(data, level, wavelet_name);

    % 2. 重构各层系数对应的子频带信号并计算ASPE
    % 初始化输出向量
    haspe_vector = zeros(1, level + 1);

    % 计算最后一层近似系数(低频)的ASPE
    approx_signal = wrcoef('a', C, L, wavelet_name, level);
    haspe_vector(1) = ASPE(approx_signal, m, t);

    % 循环计算各层细节系数(高频)的ASPE
    for i = 1:level
        detail_signal = wrcoef('d', C, L, wavelet_name, i);
        % 向量的顺序是 D_L, D_{L-1}, ..., D_1
        haspe_vector(1 + (level - i) + 1) = ASPE(detail_signal, m, t);
    end
end
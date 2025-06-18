function [aspe, prob_dist] = ASPE(X, d, tau)
    % 数据归一化
    X = X(:); % 确保输入为列向量
    X_min = min(X);
    X_max = max(X);
    if X_max == X_min
        aspe = 0;
        prob_dist = zeros(1, factorial(d));
        return;
    end
    X_normalized = (X - X_min) / (X_max - X_min);
    
    % 相空间重构
    T = length(X_normalized);
    N = T - (d - 1) * tau;
    if N < 1
        aspe = 0;
        prob_dist = zeros(1, factorial(d));
        return;
    end
    
    Y = zeros(N, d);
    for i = 1:N
        start_idx = i;
        Y(i, :) = X_normalized(start_idx : tau : start_idx + (d-1)*tau);
    end
    
    % 计算变异系数权重
    weights = zeros(N, 1);
    for i = 1:N
        y = Y(i, :);
        std_y = std(y);
        mean_y = mean(y);
        if mean_y == 0
            weights(i) = 0;
        else
            weights(i) = std_y / mean_y;
        end
    end
    
    % 生成排列模式并统计权重
    pattern_map = containers.Map('KeyType', 'double', 'ValueType', 'double');
    current_idx = 1;
    patterns = zeros(N, 1);
    
    for i = 1:N
        [~, idx] = sort(Y(i, :));
        pattern = idx2pattern(idx);
        if ~isKey(pattern_map, pattern)
            pattern_map(pattern) = current_idx;
            current_idx = current_idx + 1;
        end
        patterns(i) = pattern_map(pattern);
    end
    
    % 计算概率分布
    total_weight = sum(weights);
    if total_weight == 0
        aspe = 0;
        prob_dist = zeros(1, factorial(d));
        return;
    end
    
    pattern_weights = accumarray(patterns, weights);
    probabilities = pattern_weights / total_weight;
    
    % 处理概率为零的情况
    epsilon = 1e-10;
    probabilities(probabilities < epsilon) = epsilon;
    
    % 生成所有可能的排列模式
    all_perms = perms(1:d);
    prob_dist = zeros(1, size(all_perms, 1));
    
    % 填充概率分布
    for i = 1:size(all_perms, 1)
        perm_pattern = idx2pattern(all_perms(i, :));
        if isKey(pattern_map, perm_pattern)
            prob_dist(i) = probabilities(pattern_map(perm_pattern));
        else
            prob_dist(i) = 0;
        end
    end
    
    % 计算香农熵并归一化
    shannon_entropy = -sum(prob_dist .* log2(prob_dist));
    max_entropy = log2(factorial(d));
    aspe = shannon_entropy / max_entropy;
end

function pattern = idx2pattern(idx)
    % 将排列索引转换为唯一数值（基于阶乘权重）
    d = length(idx);
    pattern = 0;
    for i = 1:d
        pattern = pattern * d + idx(i);
    end
end
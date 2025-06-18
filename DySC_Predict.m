function predictions = DySC_Predict(model, data)
    % model: 包含训练好的参数 (params_vec) 和结构 (structure)
    % data:  测试数据结构体
    
    % 前向传播，获取模型输出
    [~, final_output] = DySC_Forward_Pass(model.params_vec, model.structure, data);
    
    % 将 softmax 的输出概率转换为类别标签
    [~, predictions] = max(final_output, [], 1);
    predictions = predictions'; % 转为列向量
end


% DySC_Forward_Pass.m (内部函数)
% 实现 DySC 模型的完整前向传播

function [hidden_fused, final_output] = DySC_Forward_Pass(params_vec, structure, data)
    
    % 1. 将扁平的参数向量恢复到结构体中
    params = struct();
    current_pos = 1;
    fields = fieldnames(structure.param_sizes);
    for k = 1:numel(fields)
        field_name = fields{k};
        sz = structure.param_sizes.(field_name);
        num_elements = prod(sz);
        params.(field_name) = reshape(params_vec(current_pos : current_pos + num_elements - 1), sz);
        current_pos = current_pos + num_elements;
    end

    % 提取各通路数据
    X_ms = data.multi_scale;
    X_h = data.hierarchical;
    X_ti = data.time_invariant;
    num_samples = size(X_ms, 1);

    % --- 2. 动态突触层 ---
    
    % a) 多尺度通路 (Scale-dependent Attention Mask)
    % mask 是可学习的，直接与输入相乘
    weighted_ms = X_ms .* params.attention_mask_ms;
    
    % b) 层次通路 (Frequency-sensitive Gating)
    % 权重由输入值和可学习的 attention 向量共同决定
    gate_h = tanh(X_h .* params.attention_vec_h); % 使用tanh作为门控函数
    weighted_h = X_h .* gate_h;
    
    % c) 时不变通路 (Stability-sensitive Gating)
    % 权重由输入向量的方差决定，gamma是可学习的
    var_ti = var(X_ti, 0, 2); % 沿行计算方差
    gate_ti = exp(-params.gamma_ti * var_ti); % 低方差 -> 高权重
    weighted_ti = X_ti .* gate_ti; % gate广播到每一列

    % --- 3. 分层信息融合 ---
    
    % a) 第一隐藏层 (通路内部融合)
    fused_ms = tanh(weighted_ms * params.W_ms + params.b_ms);
    fused_h = tanh(weighted_h * params.W_h + params.b_h);
    fused_ti = tanh(weighted_ti * params.W_ti + params.b_ti);
    
    % b) 第二隐藏层 (通路之间融合)
    combined_fused = [fused_ms, fused_h, fused_ti];
    hidden_fused = tanh(combined_fused * params.W_fusion + params.b_fusion);
    
    % --- 4. 输出层 ---
    output_logits = hidden_fused * params.W_output + params.b_output;
    
    % Softmax 激活函数
    % 为了数值稳定性，减去最大值
    exp_logits = exp(output_logits - max(output_logits, [], 2));
    final_output = (exp_logits ./ sum(exp_logits, 2))'; % 转置为 [num_classes x num_samples]
end
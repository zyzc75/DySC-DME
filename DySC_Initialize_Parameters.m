% DySC_Initialize_Parameters.m
% 初始化 DySC 模型的所有可学习参数

function model_params = DySC_Initialize_Parameters(data, num_classes)
    % data: 训练数据的一个样本，用于获取维度信息
    % num_classes: 分类的类别总数
    
    % 定义隐藏层的大小 (可根据需求调整)
    hidden_size_pathway = 16; % 通路内部融合后的维度
    hidden_size_fusion = 32;  % 通路之间融合后的维度

    % 获取输入特征维度
    dims.ms = size(data.multi_scale, 2);
    dims.h = size(data.hierarchical, 2);
    dims.ti = size(data.time_invariant, 2);

    % 存储参数的尺寸，用于后续恢复参数形状
    structure = struct();
    
    % --- 动态突触参数 ---
    structure.param_sizes.attention_mask_ms = [1, dims.ms]; % 多尺度通路的注意力掩码
    structure.param_sizes.attention_vec_h = [1, dims.h];   % 层次通路的注意力向量
    structure.param_sizes.gamma_ti = [1, 1];               % 时不变通路的方差敏感度参数

    % --- 融合层参数 ---
    % 第一隐藏层 (通路内部)
    structure.param_sizes.W_ms = [dims.ms, hidden_size_pathway];
    structure.param_sizes.b_ms = [1, hidden_size_pathway];
    structure.param_sizes.W_h = [dims.h, hidden_size_pathway];
    structure.param_sizes.b_h = [1, hidden_size_pathway];
    structure.param_sizes.W_ti = [dims.ti, hidden_size_pathway];
    structure.param_sizes.b_ti = [1, hidden_size_pathway];
    
    % 第二隐藏层 (通路之间)
    combined_dim = hidden_size_pathway * 3;
    structure.param_sizes.W_fusion = [combined_dim, hidden_size_fusion];
    structure.param_sizes.b_fusion = [1, hidden_size_fusion];
    
    % 输出层
    structure.param_sizes.W_output = [hidden_size_fusion, num_classes];
    structure.param_sizes.b_output = [1, num_classes];

    % 使用随机值初始化所有参数
    params = struct();
    fields = fieldnames(structure.param_sizes);
    for i = 1:numel(fields)
        field_name = fields{i};
        % 使用 Glorot/Xavier 初始化，有助于梯度传播
        params.(field_name) = (rand(structure.param_sizes.(field_name)) - 0.5) .* 2 .* ...
                                sqrt(6 / (sum(structure.param_sizes.(field_name))));
    end
    
    model_params.params = params;
    model_params.structure = structure;
end
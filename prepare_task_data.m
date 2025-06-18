% prepare_task_data.m
% 根据任务定义，从完整数据集中提取和组织数据的辅助函数

function [task_features, task_labels] = prepare_task_data(full_features, data_map, task_config)
    
    num_final_classes = numel(task_config);
    feature_names = fieldnames(full_features);
    
    % 初始化用于存储最终数据的细胞数组
    features_per_class = cell(num_final_classes, 1);
    labels_per_class = cell(num_final_classes, 1);

    % 遍历每一个最终的类别定义
    for i = 1:num_final_classes
        
        current_class_definition = task_config{i};
        
        % 初始化用于汇集当前类别数据的临时变量
        temp_class_features = struct();
        for fn = 1:numel(feature_names)
            temp_class_features.(feature_names{fn}) = [];
        end
        
        % 遍历定义中的每一个原始集合 (例如 'A', 'B', 'C')
        for j = 1:numel(current_class_definition)
            set_name = current_class_definition{j};
            
            if isfield(data_map, set_name)
                indices = data_map.(set_name);
                
                % 将该集合的所有特征数据追加到临时变量中
                for fn = 1:numel(feature_names)
                    feature_field = feature_names{fn};
                    temp_class_features.(feature_field) = [temp_class_features.(feature_field); full_features.(feature_field)(indices, :)];
                end
            else
                warning('在 data_map 中未找到集合: %s，已跳过。', set_name);
            end
        end
        
        % 将汇集好的数据存入细胞数组
        features_per_class{i} = temp_class_features;
        % 为这个新类别创建标签 (类别 1, 2, 3...)
        num_samples_in_class = size(temp_class_features.(feature_names{1}), 1);
        labels_per_class{i} = ones(num_samples_in_class, 1) * i;
    end
    
    % --- 最后，将所有类别的数据和标签合并成最终的输出 ---
    task_features = struct();
    for fn = 1:numel(feature_names)
        task_features.(feature_names{fn}) = [];
    end
    task_labels = [];
    
    for i = 1:num_final_classes
        for fn = 1:numel(feature_names)
            feature_field = feature_names{fn};
            task_features.(feature_field) = [task_features.(feature_field); features_per_class{i}.(feature_field)];
        end
        task_labels = [task_labels; labels_per_class{i}];
    end
end
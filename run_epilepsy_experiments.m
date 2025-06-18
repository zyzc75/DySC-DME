% clear; clc; close all;

%% 步骤 1: 加载您的完整数据集

fprintf('正在加载数据...\n');
load features_st.mat;
% 假设前100个是A(1), 接着100个是B(2), ..., 最后100个是E(5)
labels_original = repelem((1:5)', 100, 1);
fprintf('数据加载完成。\n');

%% 步骤 2: 定义原始数据集的结构
% 根据您的数据情况，定义A,B,C,D,E五个集合对应的行索引
% (这里假设每个集合100行)
data_map.A = 1:100;
data_map.B = 101:200;
data_map.C = 201:300;
data_map.D = 301:400;
data_map.E = 401:500;

%% ========================================================================
%                     用户控制面板: 在此定义您的分类任务
% ========================================================================
%  任务定义规则:
% --- 示例 1: A vs E (经典的 健康 vs 癫痫发作) ---
task_config = {{'C'}, {'E'}};
task_name = '健康 (AB) vs CD vs癫痫发作 (E)';
% ========================================================================
%                           任务定义结束
% ========================================================================

% %% 步骤 4: 根据任务定义，准备数据
fprintf('正在为任务 "%s" 准备数据...\n', task_name);
[task_features, task_labels] = prepare_task_data(features_st, data_map, task_config);
%% 步骤 5: 选择并运行分类器
% 您可以取消注释想运行的分类器

% % % --- 运行我们设计的 DySC 分类器 ---
fprintf('\n>>>>>> 正在为任务 "%s" 运行 DySC 分类器 <<<<<<\n', task_name);
% DySC_Train_Test(task_features, task_labels, 10); % 使用10折交叉验证

% fprintf('数据准备完成. 共 %d 个样本, %d 个类别。\n', numel(task_labels), numel(unique(task_labels)));
fprintf("%.2f(±%.2f)\n",ans.mean_accuracy*100,ans.std_accuracy*100);
fprintf("%.2f(±%.2f)\n",ans.mean_precision*100,ans.std_precision*100);
fprintf("%.2f(±%.2f)\n",ans.mean_recall*100,ans.std_recall*100);
fprintf("%.2f(±%.2f)\n",ans.mean_f1_score*100,ans.std_f1_score*100);
fprintf("%.2f(±%.2f)\n",ans.mean_specificity*100,ans.std_specificity*100);

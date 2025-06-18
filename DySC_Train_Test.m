% DySC_Train_Test.m (版本 6.0: 无早停，带标准差)
% 动态突触分类器 (DySC) 的主训练与测试函数
% 训练至自然收敛，并报告完整的性能指标(含标准差)

function results = DySC_Train_Test(features, labels, N)
    fprintf('--- 开始动态突触分类器 (DySC) 的 %d-折交叉验证 ---\n', N);

    % --- 数据准备 ---
    multi_scale_features = [features.aspe, features.rcmaspe];
    hierarchical_features = features.haspe;
    time_invariant_features = features.tsmaspe;
    
    unique_labels = unique(labels);
    num_samples = size(labels, 1);
    num_classes = numel(unique_labels);
    one_hot_labels = full(ind2vec(labels', num_classes));

    % --- 初始化用于存储每一折性能的数组 ---
    fold_metrics = struct();
    fold_metrics.accuracy     = zeros(N, 1);
    fold_metrics.precision    = zeros(N, 1);
    fold_metrics.recall       = zeros(N, 1);
    fold_metrics.f1_score     = zeros(N, 1);
    fold_metrics.specificity  = zeros(N, 1);

    % --- N-折交叉验证设置 ---
    cv = cvpartition(num_samples, 'KFold', N);
    all_predictions = zeros(num_samples, 1);
    all_true_labels = zeros(num_samples, 1);
    
    % 初始化一个结构体来存储最后一折的收敛历史
    training_history = struct('fval', [], 'firstorderopt', []);

    for i = 1:N
        fprintf('\n--- 正在处理第 %d / %d 折 ---\n', i, N);

        % --- 数据划分 (无验证集) ---
        train_idx = find(training(cv, i));
        test_idx = find(test(cv, i));

        % 准备训练集数据
        train_data.multi_scale = multi_scale_features(train_idx, :);
        train_data.hierarchical = hierarchical_features(train_idx, :);
        train_data.time_invariant = time_invariant_features(train_idx, :);
        train_labels_one_hot = one_hot_labels(:, train_idx);
        
        % --- 模型训练 (无早停) ---
        fprintf('正在训练模型 (直至自然收敛)...\n');
        model_params = DySC_Initialize_Parameters(train_data, num_classes);
        
        % 如果是最后一折，清空历史记录，准备记录
        if i == N
            training_history.fval = [];
            training_history.firstorderopt = [];
        end

        options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
                               'Display', 'iter', 'MaxIterations', 100, ...
                               'MaxFunctionEvaluations', 1000000, ...
                               'StepTolerance', 1e-6, ...
                               'OutputFcn', @recordHistory); % 使用简单的历史记录函数
        
        loss_function = @(p) DySC_Loss(p, model_params.structure, train_data, train_labels_one_hot);
        initial_params_vec = DySC_Flatten_Parameters(model_params);
        [trained_params_vec, ~] = fminunc(loss_function, initial_params_vec, options);
        
        trained_model.params_vec = trained_params_vec;
        trained_model.structure = model_params.structure;
        fprintf('模型训练完成。\n');

        % --- 模型测试 ---
        fprintf('正在测试模型...\n');
        test_data.multi_scale = multi_scale_features(test_idx, :);
        test_data.hierarchical = hierarchical_features(test_idx, :);
        test_data.time_invariant = time_invariant_features(test_idx, :);
        predictions = DySC_Predict(trained_model, test_data);
        all_predictions(test_idx) = predictions;
        all_true_labels(test_idx) = labels(test_idx);
        
        % --- 在每一折内部计算性能指标 ---
        C_fold = confusionmat(labels(test_idx), predictions);
        fold_metrics.accuracy(i) = sum(diag(C_fold)) / sum(C_fold(:));

        per_class_metrics = struct('precision', zeros(num_classes,1), 'recall', zeros(num_classes,1), 'specificity', zeros(num_classes,1), 'f1_score', zeros(num_classes,1));
        for k = 1:num_classes
            TP = C_fold(k,k); FP = sum(C_fold(:,k)) - TP; FN = sum(C_fold(k,:)) - TP; TN = sum(C_fold(:)) - TP - FP - FN;
            per_class_metrics.precision(k) = TP / (TP + FP);
            per_class_metrics.recall(k) = TP / (TP + FN);
            per_class_metrics.specificity(k) = TN / (TN + FP);
            per_class_metrics.f1_score(k) = 2 * per_class_metrics.precision(k) * per_class_metrics.recall(k) / (per_class_metrics.precision(k) + per_class_metrics.recall(k));
        end
        per_class_metrics.precision(isnan(per_class_metrics.precision)) = 0; per_class_metrics.recall(isnan(per_class_metrics.recall)) = 0;
        per_class_metrics.specificity(isnan(per_class_metrics.specificity)) = 0; per_class_metrics.f1_score(isnan(per_class_metrics.f1_score)) = 0;

        fold_metrics.precision(i) = mean(per_class_metrics.precision); fold_metrics.recall(i) = mean(per_class_metrics.recall);
        fold_metrics.specificity(i) = mean(per_class_metrics.specificity); fold_metrics.f1_score(i) = mean(per_class_metrics.f1_score);
        
        fprintf('第 %d 折性能: Acc=%.4f, F1=%.4f\n', i, fold_metrics.accuracy(i), fold_metrics.f1_score(i));
    end

    % --- 计算并显示所有折的平均值和标准差 ---
    fprintf('\n--- 交叉验证完成，正在计算最终性能指标 ---\n');
    
    results.mean_accuracy = mean(fold_metrics.accuracy); results.std_accuracy = std(fold_metrics.accuracy);
    results.mean_precision = mean(fold_metrics.precision); results.std_precision = std(fold_metrics.precision);
    results.mean_recall = mean(fold_metrics.recall); results.std_recall = std(fold_metrics.recall);
    results.mean_f1_score = mean(fold_metrics.f1_score); results.std_f1_score = std(fold_metrics.f1_score);
    results.mean_specificity = mean(fold_metrics.specificity); results.std_specificity = std(fold_metrics.specificity);
    results.fold_metrics = fold_metrics;
    
    fprintf('\n----- 总体性能评估 (平均值 ± 标准差) -----\n');
    fprintf('分类准确率 (Accuracy)   : %.4f (± %.4f)\n', results.mean_accuracy, results.std_accuracy);
    fprintf('宏平均精度 (Precision)    : %.4f (± %.4f)\n', results.mean_precision, results.std_precision);
    fprintf('宏平均召回率 (Recall)     : %.4f (± %.4f)\n', results.mean_recall, results.std_recall);
    fprintf('宏平均F1分数 (F1-Score)   : %.4f (± %.4f)\n', results.mean_f1_score, results.std_f1_score);
    fprintf('宏平均特异度 (Specificity): %.4f (± %.4f)\n', results.mean_specificity, results.std_specificity);
    fprintf('-------------------------------------------\n');
    
    % --- 可视化 ---
    figure;
    cm = confusionchart(all_true_labels, all_predictions);
    cm.Title = sprintf('%d-Fold Cross-Validation Confusion Matrix (Overall)', N);
    cm.RowSummary = 'row-normalized'; cm.ColumnSummary = 'column-normalized';
    results.overall_confusion_matrix = cm.NormalizedValues;
    
    fprintf('正在绘制最后一折的收敛过程图...\n');
    figure;
    yyaxis left;
    semilogy(training_history.fval, '-o', 'LineWidth', 1.5, 'Color', [0, 0.4470, 0.7410]);
    ylabel('损失函数值 f(x) (对数尺度)');
    xlabel('迭代次数 (Iteration)');
    title('模型训练收敛过程 (最后一折)');
    grid on;
    yyaxis right;
    semilogy(training_history.firstorderopt, '--s', 'LineWidth', 1.5, 'Color', [0.8500, 0.3250, 0.0980]);
    ylabel('一阶最优性 (对数尺度)');
    legend('损失函数值', '一阶最优性', 'Location', 'northeast');
    ax = gca; ax.YAxis(1).Color = [0, 0.4470, 0.7410]; ax.YAxis(2).Color = [0.8500, 0.3250, 0.0980];

    % --- 嵌套的历史记录函数 ---
    function stop = recordHistory(~, optimValues, state)
        stop = false; 
        if strcmp(state, 'iter') && i == N
            training_history.fval = [training_history.fval; optimValues.fval];
            training_history.firstorderopt = [training_history.firstorderopt; optimValues.firstorderopt];
        end
    end
end

% --- 文件末尾的辅助函数 ---
function params_vec = DySC_Flatten_Parameters(model_params)
    params_vec = []; fields = fieldnames(model_params.params);
    for k = 1:numel(fields), field_name = fields{k}; params_vec = [params_vec; model_params.params.(field_name)(:)]; end
end
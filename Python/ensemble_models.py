import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def final_meta_model(ensemble_model):
    if ensemble_model == 'gradient_boosting':
        model = GradientBoostingClassifier()
    elif ensemble_model == 'random_forest':
        model = RandomForestClassifier()
    elif ensemble_model == 'xgboost':
        model = XGBClassifier()
    elif ensemble_model == 'lightgbm':
        model = LGBMClassifier()
    elif ensemble_model == 'adaboost':
        model = AdaBoostClassifier()
    elif ensemble_model == 'extra_trees':
        model = ExtraTreesClassifier()
    elif ensemble_model == 'svm':
        model = SVC()
    else:
        raise ValueError('Invalid ensemble model name.')

    return model

'''
GBM = GradientBoostingClassifier(
    loss='deviance',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=1.0,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    random_state=None
)
'''

'''
RandomForest = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='auto',
    bootstrap=True,
    random_state=None
)
'''

'''
XGBoost = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=1.0,
    colsample_bytree=1.0,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    random_state=None
)
'''

'''
LightGBM = LGBMClassifier(
    boosting_type='gbdt',
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.1,
    n_estimators=100,
    subsample_for_bin=200000,
    objective=None,
    class_weight=None,
    min_split_gain=0.0,
    min_child_weight=0.001,
    min_child_samples=20,
    subsample=1.0,
    subsample_freq=0,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=None,
    n_jobs=-1,
    silent=True,
    importance_type='split'
)
'''

'''
CatBoost = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    verbose=0,
    random_seed=None,
    task_type='CPU',
    devices=None,
    eval_metric=None,
    custom_metric=None,
    use_best_model=None,
    early_stopping_rounds=None,
    save_snapshot=None,
    snapshot_file=None,
    snapshot_interval=None,
    fold_permutation_block=None,
    border_count=None,
    feature_border_type=None,
    per_float_feature_quantization=None,
    l2_leaf_reg=None,
    model_size_reg=None,
    max_ctr_complexity=None,
    ctr_leaf_count_limit=None,
    ctr_target_border_count=None,
    ctr_target_border=None,
    ctr_border_count=None,
    ctr_data=None,
    ctr_leaf_count_mode=None,
    nan_mode=None,
    counter_calc_method=None,
    leaf_estimation_iterations=None,
    leaf_estimation_method=None,
    thread_count=None,
    random_strength=None,
    bagging_temperature=None,
    od_pval=None,
    od_wait=None,
    od_type=None,
    training_border=None,
    training_border_random_seed=None,
    allow_writing_files=None,
    final_ctr_computation_mode=None,
    approx_on_full_history=None,
    boosting_type=None,
    simple_ctr=None,
    combinations_ctr=None,
    per_feature_ctr=None,
    ctr_description=None,
    task_type="CPU",
    bootstrap_type=None,
    subsample=None,
    sampling_frequency=None,
    one_hot_max_size=None,
    random_seed=None,
    depth=None,
    ctr_target=None,
    rsm=None,
    boost_from_average=None,
    model_shrink_rate=None,
    model_shrink_mode=None,
    langevin=None,
    diffusion_temperature=None,
    feature_calcers=None,
    metadata=None,
    leaf_estimation_backtracking=None,
    best_model_min_trees=None,
    best_model_min_trees_rate=None,
    max_ctr_complexity=None,
    ignored_features=None,
    leaf_estimation_iterations=None,
    fold_len_multiplier=None,
    approx_on_full_history=None,
    boost_from_average=None,
    subsample=None,
    use_best_model=None,
    gpu_cat_features_storage=None,
    data_partition=None,
    od_pval=None,
    od_wait=None,
    od_type=None,
    counter_calc_method=None,
    leaf_estimation_method=None,
    thread_count=None,
    random_strength=None,
    bagging_temperature=None,
    save_snapshot=None,
    snapshot_file=None,
    snapshot_interval=None,
    fold_permutation_block=None,
    devices=None,
    leaf_estimation_backtracking=None,
    use_best_model=None,
    best_model_min_trees=None,
    best_model_min_trees_rate=None,
    approx_on_full_history=None,
    boost_from_average=None,
    subsample=None,
    gpu_ram_part=None,
    pinned_memory_size=None,
    gpu_cat_features_storage=None,
    data_partition=None,
    metadata=None,
    early_stopping_rounds=None,
    verbose=None
)
'''

'''
AdaBoost = AdaBoostClassifier(
    base_estimator=None,
    n_estimators=50,
    learning_rate=1.0,
    algorithm='SAMME.R',
    random_state=None
)
'''

'''
ExtraTrees = ExtraTreesClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='auto',
    random_state=None
)
'''

'''
SVM = SVC(
    C=1.0,
    kernel='rbf',
    degree=3,
    gamma='scale',
    coef0=0.0,
    shrinking=True,
    probability=False,
    tol=0.001,
    cache_size=200,
    class_weight=None,
    verbose=False,
    max_iter=-1,
    decision_function_shape='ovr',
    break_ties=False,
    random_state=None
)
'''

'''
LogisticRegression = LogisticRegression(
    penalty='l2',
    dual=False,
    tol=0.0001,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=None,
    solver='lbfgs',
    max_iter=100,
    multi_class='auto',
    verbose=0,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None
)
'''



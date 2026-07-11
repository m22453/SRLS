#!/bin/sh

# 定义 beta 的取值范围
beta_values="0.001"

# 定义 train_mode 的取值范围
train_modes="sf"

# 定义 missing_rate 的取值范围
missing_rates="0.1 0.3 0.5 0.7 0.9"
# fea_selection_rates="0.6 0.7 0.8 0.9 1.0"
# n_hetero_layers="1 3 4"

# 遍历 train_mode、beta 和 fea_selection_rate 的每个值
for train_mode in $train_modes; do
    for beta in $beta_values; do
        for missing_rate in $missing_rates; do
            echo "Running with train_mode=$train_mode, beta=$beta, and missing_rate=$missing_rate"
            python main_LU21.py \
                --beta $beta \
                --train_mode $train_mode \
                --missing_rate $missing_rate
        done
    done
done
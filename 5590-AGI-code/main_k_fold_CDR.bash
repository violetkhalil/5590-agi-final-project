k_fold=5
epoch=200
task=CDR
num_classes=2

lr=0.1
alpha_eta_product=0.9
rho_coeff=20

n_MLP_layers=1
n_GNN_layers=2
pooling="sum"

PYTHON=/home/heng/anaconda3/envs/IBGNN/bin/python
mkdir -p logs

# ============================================================
# Helper functions
# ============================================================
run_baselines() {
    local data=$1 cdr_pair=$2 gpu=$3
    local epsilon=$4 rho=$5
    local batch_size=$6 hidden_dim=$7 dropout=$8 seed=$9

    CUDA_VISIBLE_DEVICES=$gpu $PYTHON main_k_fold_CDR.py \
        --data $data --k_fold $k_fold --task $task --num_classes $num_classes \
        --cdr_pair $cdr_pair \
        --optimizer Basic_GNN_CDR \
        --batch_size $batch_size --num_epochs $epoch \
        --lr $lr --epsilon $epsilon --rho $rho \
        --alpha_eta_product $alpha_eta_product \
        --hidden_dim $hidden_dim --dropout $dropout \
        --n_MLP_layers $n_MLP_layers --n_GNN_layers $n_GNN_layers \
        --pooling $pooling --manual_seed $seed

    CUDA_VISIBLE_DEVICES=$gpu $PYTHON main_k_fold_CDR.py \
        --data $data --k_fold $k_fold --task $task --num_classes $num_classes \
        --cdr_pair $cdr_pair \
        --optimizer BrainNNExplainer_CDR \
        --batch_size $batch_size --num_epochs $epoch \
        --lr $lr --epsilon $epsilon --rho $rho \
        --alpha_eta_product $alpha_eta_product \
        --hidden_dim $hidden_dim --dropout $dropout \
        --n_MLP_layers $n_MLP_layers --n_GNN_layers $n_GNN_layers \
        --pooling $pooling --manual_seed $seed
}

run_dsvrbgd() {
    local data=$1 cdr_pair=$2 gpu=$3
    local epsilon=$4 rho=$5
    local batch_size=$6 hidden_dim=$7 dropout=$8 seed=$9
    local eta_x_expand=${10}

    CUDA_VISIBLE_DEVICES=$gpu $PYTHON main_k_fold_CDR.py \
        --data $data --k_fold $k_fold --task $task --num_classes $num_classes \
        --cdr_pair $cdr_pair \
        --optimizer FO-DSVRBGD_CDR \
        --eta_x_expand $eta_x_expand \
        --batch_size $batch_size --num_epochs $epoch \
        --lr $lr --epsilon $epsilon --rho $rho \
        --alpha_eta_product $alpha_eta_product \
        --rho_coeff $rho_coeff \
        --hidden_dim $hidden_dim --dropout $dropout \
        --n_MLP_layers $n_MLP_layers --n_GNN_layers $n_GNN_layers \
        --pooling $pooling --manual_seed $seed
}

# ============================================================
# 每个 (data, cdr_pair) 在独立子 shell 中后台运行，各占一张 GPU
# ============================================================

# ADNI — NC vs SMC  (gpu=0)
(
    data=ADNI; cdr_pair=NC_SMC; gpu=0
    batch_size=8; hidden_dim=16; dropout=0.2; seed=42; eta_x_expand=100
    epsilon_l_dsvrbgd=(0.08); rho=5
    # for epsilon in 0.1; do run_baselines $data $cdr_pair $gpu $epsilon $rho $batch_size $hidden_dim $dropout $seed; done
    for epsilon in "${epsilon_l_dsvrbgd[@]}"; do
        run_dsvrbgd $data $cdr_pair $gpu $epsilon $rho $batch_size $hidden_dim $dropout $seed $eta_x_expand
    done
) > logs/CDR_ADNI_NC_SMC.log 2>&1 &

# ADNI — SMC vs MCI  (gpu=1)
(
    data=ADNI; cdr_pair=SMC_MCI; gpu=1
    batch_size=8; hidden_dim=16; dropout=0.2; seed=42; eta_x_expand=100
    epsilon_l_dsvrbgd=(0.08); rho=5
    # for epsilon in 0.1; do run_baselines $data $cdr_pair $gpu $epsilon $rho $batch_size $hidden_dim $dropout $seed; done
    for epsilon in "${epsilon_l_dsvrbgd[@]}"; do
        run_dsvrbgd $data $cdr_pair $gpu $epsilon $rho $batch_size $hidden_dim $dropout $seed $eta_x_expand
    done
) > logs/CDR_ADNI_SMC_MCI.log 2>&1 &

# OASIS — NC vs MCI  (gpu=2)
(
    data=OASIS; cdr_pair=NC_MCI; gpu=2
    batch_size=16; hidden_dim=20; dropout=0.3; seed=42; eta_x_expand=100
    epsilon_l_dsvrbgd=(0.08); rho=5
    # for epsilon in 0.1; do run_baselines $data $cdr_pair $gpu $epsilon $rho $batch_size $hidden_dim $dropout $seed; done
    for epsilon in "${epsilon_l_dsvrbgd[@]}"; do
        run_dsvrbgd $data $cdr_pair $gpu $epsilon $rho $batch_size $hidden_dim $dropout $seed $eta_x_expand
    done
) > logs/CDR_OASIS_NC_MCI.log 2>&1 &

# OASIS — MCI vs AD  (gpu=3, uncomment to enable)
# (
#     data=OASIS; cdr_pair=MCI_AD; gpu=3
#     batch_size=16; hidden_dim=20; dropout=0.3; seed=42; eta_x_expand=100
#     epsilon_l_dsvrbgd=(0.08); rho=5
#     for epsilon in "${epsilon_l_dsvrbgd[@]}"; do
#         run_dsvrbgd $data $cdr_pair $gpu $epsilon $rho $batch_size $hidden_dim $dropout $seed $eta_x_expand
#     done
# ) &

wait
echo "All CDR experiments finished."

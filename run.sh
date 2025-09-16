gpu_id=0
master_port=8901
config_path=(
    # ./config/dermamnist_resnet18_w2a2.hocon
    # ./config/dermamnist_resnet18_w3a3.hocon
    # ./config/dermamnist_resnet18_w4a4.hocon
    # ./config/tissuemnist_resnet18_w2a2.hocon
    # ./config/tissuemnist_resnet18_w3a3.hocon
    # ./config/tissuemnist_resnet18_w4a4.hocon
    ./config/bloodmnist_resnet18_w2a2.hocon
    ./config/bloodmnist_resnet18_w3a3.hocon
    # ./config/bloodmnist_resnet18_w4a4.hocon
    ./config/pathmnist_resnet18_w2a2.hocon
    ./config/pathmnist_resnet18_w3a3.hocon
    # ./config/pathmnist_resnet18_w4a4.hocon
)

for config in "${config_path[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=${master_port} main_direct.py --conf_path ${config}
done

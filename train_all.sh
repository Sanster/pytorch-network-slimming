models="vgg11_bn resnet18 resnet50 RepVGG-A0-woid"

# exit when any command fails
set -e

epochs=120
batch_size=64
learning_rate=0.01
ss="0.0001"
prune_ratios="0.5 0.75"
fine_tune_epochs=120
fine_tune_learning_rate=0.01
output_dir="./output"


for net in $models; do
  for s in $ss; do
    save_dir=${output_dir}/"${net}_s_${s}"
    python3 train.py \
      --save_dir $save_dir \
      --net $net \
      --epoch $epochs \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --sparsity_train \
      --s $s

    for prune_ratio in $prune_ratios; do
      python3 train.py \
        --save_dir ${save_dir} \
        --net $net \
        --batch_size $batch_size \
        --ckpt ${save_dir}/last.ckpt \
        --fine_tune \
        --prune_schema ./schema/$net.json \
        --fine_tune_epoch $fine_tune_epochs \
        --fine_tune_learning_rate $fine_tune_learning_rate \
        --prune_ratio $prune_ratio \
        --s $s
    done
  done
done

python3 summary.py
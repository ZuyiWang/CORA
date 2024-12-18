name=$0
. configs/controller.sh

# --batch_size 4 \
# --num_queries 1000 \
# --dec_layers 6 \
# --use_gradient_accumulation \
# --accumulation_steps 4 \
# --use_deformable_attention
# --use_dino
# --epochs 35 \
args=" \
--coco_path $data_path \
--output_dir $work_dir \
--batch_size 4 \
--epochs 30 \
--lr_drop 35 \
--backbone clip_RN50 \
--text_len 15 \
--ovd \
--region_prompt_path logs/region_prompt_R50.pth \
--save_every_epoch 50 \
--dim_feedforward 1024 \
--use_nms \
--num_queries 1000 \
--anchor_pre_matching \
--remove_misclassified \
--condition_on_text \
--enc_layers 6 \
--text_dim 1024 \
--condition_bottleneck 128 \
--split_class_p 0.2 \
--model_ema \
--model_ema_decay 0.99996 \
--save_best \
--label_version RN50base \
--disable_init \
--target_class_factor 8 \
--use_dino \
--dino_cfg /home/wangzy/ObjectDetectionModel/CORA/configs/ovdino.py \
--amp \
"

eval "$header$args$extra_args 2>&1 | tee -a $work_dir/exp_$now.txt"
####### for pdb debug
# eval "$header$args$extra_args"

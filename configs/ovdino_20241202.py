###### copy from DINO https://github.com/IDEA-Research/DINO DINO_5scale.py
###### modify for OV DINO

###### OV DQUO ######
##### start DINO parameters ##### 
param_dict_type = "default"
lr_backbone_names = ["backbone.0"]
lr_linear_proj_names = ["reference_points", "sampling_offsets"] # different!!!
lr_linear_proj_mult = 0.1
ddetr_lr_param = False
weight_decay = 0.0001
clip_max_norm = 0.1
use_checkpoint = False
position_embedding = "sine"
pe_temperatureH = 20 # pe_temperature=20 in main.py, used in position_embedding
pe_temperatureW = 20
enc_layers = 6
dec_layers = 6
unic_layers = 0
pre_norm = False
dim_feedforward = 2048 # different!!!
hidden_dim = 256
dropout = 0.0 # different!!!
nheads = 8
num_queries =1000
query_dim = 4
num_patterns = 0
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
random_refpoints_xy = False
fix_refpoints_hw = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False
use_deformable_box_attn = False
box_attn_type = "roi_align"
dec_layer_number = None
enc_n_points = 4  
dec_n_points = 4
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = False
add_pos_value = False
two_stage_type = "standard"
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
transformer_activation = "relu"
batch_norm_type = "FrozenBatchNorm2d"
masks = False
aux_loss = True
dec_pred_bbox_embed_share = False # different!!!
dec_pred_class_embed_share = True
use_detached_boxes_dec_out = False 
##### end DINO parameters ##### 


##### start loss parameters ##### 
set_cost_class = 2.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 2.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25
decoder_sa_type = "sa"  # ['sa', 'ca_label', 'ca_content']
decoder_module_seq = ["sa", "ca", "ffn"]
##### end loss parameters ##### 


##### start dn parameters ##### 
use_dn = True
dn_number = 100
dn_box_noise_scale = 1.0 # different!!!
dn_label_noise_ratio = 0.5
embed_init_tgt = True
##### end dn parameters ##### 


##### start ema parameters ##### 
use_ema = True # setting in train shell
ema_decay = 0.99996
ema_epoch = 0
##### end ema parameters ##### 


##### start open-vocabulary training parameters ##### 
lr = 1e-4
epochs = 30
lr_drop = 50
batch_size = 4
lr_backbone=0.0
save_checkpoint_interval = 1
# wildcard="object" # not used
num_feature_levels = 3 # for OVD layer 234
# modelname = "ov_dquo" # not used
text_dim=1024  # 1024 for RN50 
# backbone = "CLIP_RN50" # setting in train shell
text_len = 15
pretrained=""
# region_prompt_path = "pretrained/region_prompt_R50.pth" # setting in train shell
# backbone_out_indice=[1, 2, 3] # C3, C4, C5 # not used
# pseudo_box = "ow_labels/OW_COCO_R2.json" # not used
in_channel=[512, 1024]
##### end open-vocabulary training parameters ##### 


##### start inference parameters ##### 
eval_tau = 100
objectness_alpha = 1.0
nms_iou_threshold = 0.5
target_class_factor=3.0 # different!!!
##### end inference parameters ##### 


##### start dataset parameters ##### 
# coco_path = "data" # setting in train shell
dataset_file = "coco"
# label_version = "standard" # setting in train shell
repeat_factor_sampling=False
dn_labelbook_size=48 # keep compatible with ovlvis, dn_labelbook_size in DINO 
# num_label_sampled=48 # keep compatible with ovlvis, dn_labelbook_size in DINO 
##### end dataset parameters ##### 

###### start CORA setting ######
# 修改在backbone.py
backbone_feature = 'layer234'




########### wzy ###################
# decoder_layer_noise = False
# use_detached_boxes_dec_out = False
# unic_layers = 0
# pre_norm = False
# query_dim = 4
# transformer_activation = 'relu'
# num_patterns = 0
# num_feature_levels = 3 # for OVD
# enc_n_points = 4
# dec_n_points = 4
# use_deformable_box_attn = False
# box_attn_type = 'roi_align'
# add_channel_attention = False
# add_pos_value = False
# random_refpoints_xy = False
# nheads = 8 # set in args
# fix_refpoints_hw = -1

# two_stage_type = 'standard'
# two_stage_pat_embed = 0
# two_stage_add_query_num = 0
# two_stage_bbox_embed_share = False
# two_stage_class_embed_share = False
# two_stage_learn_wh = False
# two_stage_default_hw = 0.05
# two_stage_keep_all_tokens = False
# num_select = 300 # used in PostProcess, now OVPostProcess

# dec_layer_number = None
# decoder_sa_type = 'sa' # ['sa', 'ca_label', 'ca_content']
# decoder_module_seq = ['sa', 'ca', 'ffn']
# embed_init_tgt = True

# for dn
# use_dn = True
# dn_number = 100
# dn_box_noise_scale = 0.4 # different from OV DQUO!!!
# dn_label_noise_ratio = 0.5
# match_unstable_error = True # deprecated
# dn_labelbook_size = 80 # for COCO, 似乎没必要80 与训练时的类别数量有关

# dec_pred_bbox_embed_share = True # different from OV DQUO!!!
# dec_pred_class_embed_share = True

# batch_norm_type = 'FrozenBatchNorm2d'
# aux_loss = True

# mask_loss_coef = 1.0 # for masks, not used
# dice_loss_coef = 1.0 # for masks, not used

# ##### start loss parameters ##### 
# set_cost_class = 2.0
# set_cost_bbox = 5.0
# set_cost_giou = 2.0
# cls_loss_coef = 2.0
# bbox_loss_coef = 5.0
# giou_loss_coef = 2.0
# enc_loss_coef = 1.0
# interm_loss_coef = 1.0
# no_interm_box_loss = False
# focal_alpha = 0.25
# decoder_sa_type = "sa"  # ['sa', 'ca_label', 'ca_content']
# decoder_module_seq = ["sa", "ca", "ffn"]

# nms_iou_threshold = -1 # ori in DINO
# nms_iou_threshold = 0.5

# # 修改在backbone.py
# backbone_feature = 'layer234'

# enc_layers = 6
# dec_layers = 6


################## Not use ##################################
# _base_ = ['coco_transformer.py']

# num_queries = 1000
# num_classes=91

# lr = 0.0001
# param_dict_type = 'default'
# lr_backbone = 1e-05
# lr_backbone_names = ['backbone.0']
# lr_linear_proj_names = ['reference_points', 'sampling_offsets']
# lr_linear_proj_mult = 0.1
# ddetr_lr_param = False
# batch_size = 1
# weight_decay = 0.0001
# epochs = 12
# lr_drop = 11
# save_checkpoint_interval = 1
# clip_max_norm = 0.1
# onecyclelr = False
# multi_step_lr = False
# lr_drop_list = [33, 45]


# modelname = 'dino'
# frozen_weights = None
# backbone = 'resnet50'
# use_checkpoint = False

# dilation = False
# position_embedding = 'sine'
# pe_temperatureH = 20
# pe_temperatureW = 20
# return_interm_indices = [0, 1, 2, 3]
# backbone_freeze_keywords = None



# dim_feedforward = 2048
# hidden_dim = 256
# dropout = 0.0




# pdetr3_bbox_embed_diff_each_layer = False
# pdetr3_refHW = -1


# dabdetr_yolo_like_anchor_update = False
# dabdetr_deformable_encoder = False
# dabdetr_deformable_decoder = False


# dec_layer_number = None
# num_feature_levels = 5


# dln_xy_noise = 0.2
# dln_hw_noise = 0.2

# matcher_type = 'HungarianMatcher' # or SimpleMinsumMatcher


# # for ema
# use_ema = False
# ema_decay = 0.9997
# ema_epoch = 0
# masks = False


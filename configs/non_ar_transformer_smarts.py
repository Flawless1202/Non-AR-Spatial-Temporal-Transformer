name = "NonARTransformerSMARTS"
version = "1.0.0"
model = dict(
    type='NonAutoRegressionTransformer',
    # vision_encoder=dict(
    #     type="VisionEncoder",
    #     vit_model_name="vit_small_resnet26d_custom",
    #     pretrained=False,
    #     in_channels=2,
    #     img_size=32,
    #     embed_dim=32),
    box_encoder=dict(
        type='MLP',
        d_in=4,
        d_hid=128,
        d_out=64,
        num_layers=1,
        activation='relu',
        norm=False,
        dropout=0),
    index_pos_embed=dict(
        type='TimeEmbeddingSine',
        d_model=64,
        temperature=10000,
        requires_grad=False),
    encoder=dict(
        type='SpatialTemporalEncoder',
        num_head=1,
        d_model=64,
        d_hid=128,
        num_stack=4,
        activation='relu',
        norm_before=True,
        embed_only_first=True,
        with_global_conv=True),
    decoder=dict(
        type='SpatialTemporalDecoder',
        num_head=1,
        d_model=64,
        d_hid=128,
        num_stack=4,
        activation='relu',
        dropout=0.1,
        norm_before=True,
        embed_only_first=True,
        with_global_conv=True),
    loss=dict(
        type='SmoothL1Loss'))
dataset_type = 'SMARTS'
data_root = "data/smarts"
pipelines = [
    dict(type='ToTensor'),
    dict(type='BoxesClip', box_range=(512, 320)),
    dict(type='BoxesNormalize', box_range=(512, 320)),
    dict(type='AlignTime', align_type='current')]
data = dict(
    batch_size=16,
    num_workers=4,
    train=dict(
        type=dataset_type,
        root_dir=data_root,
        step=3,
        hist_len=30,
        futr_len=45,
        image_size=(1280, 960),
        resize=(512, 320),
        scenes_list_file=f"{data_root}/train_sequences_list.txt",
        pipelines=pipelines),
    val=dict(
        type=dataset_type,
        root_dir=data_root,
        step=3,
        hist_len=30,
        futr_len=45,
        image_size=(1280, 960),
        resize=(512, 320),
        scenes_list_file=f"{data_root}/val_sequences_list.txt",
        pipelines=pipelines))
# training and testing settings
train_cfg = dict(
    forward=dict(mode='one_step'),
    loss=dict(box_range=(512, 320)))
val_cfg = dict(
    forward=dict(mode='one_step'),
    loss=dict(box_range=(512, 320)),
    predict=dict(pred_fmt='xyxy', box_range=(512, 320)),
    target=dict(pred_fmt='xyxy', box_range=(512, 320)))
test_cfg = dict()
# optimizer
optimizer_cfg = dict(
    type='Adam',
    lr=.1,
    betas=(0.9, 0.999),
    # eps=1e-9
    # weight_decay=1e-3
)
lr_cfg = dict(
    type="StepLR",
    step_size=50,
    gamma=.7)
warm_up_cfg = dict(
    type="Exponential",
    step_size=5000)
random_seed = 123456
num_gpus = 1
max_epochs = 600
gradient_clip_val = 0.5
checkpoint_path = "work_dirs/checkpoints"
log_path = "work_dirs/logs"
result_path = "work_dirs/results"
load_from_checkpoint = None
resume_from_checkpoint = None
test_checkpoint = "work_dirs/checkpoints/NonARTransformerSMARTS/1.0.0/last.ckpt"
batch_size_times = 1
simple_profiler = True
check_val_every_n_epoch = 1

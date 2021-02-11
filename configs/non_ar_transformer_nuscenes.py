name = "NonARTransformerNuScenes"
version = "1.0.0"
model = dict(
    type='NonAutoRegressionTransformer',
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
dataset_type = 'NuScenes'
data_root = "data/nuscenes_sequences"
pipelines = [
    dict(type='ToTensor'),
    dict(type='BoxesClip', box_range=(512, 320)),
    dict(type='BoxesNormalize', box_range=(512, 320)),
    dict(type='AlignTime', align_type='current')]
data = dict(
    batch_size=64,
    num_workers=16,
    train=dict(
        type=dataset_type,
        root_dir=data_root,
        step=1,
        hist_len=4,
        futr_len=6,
        roi_size=(32, 32, 2),
        image_size=(1600, 900),
        resize=(512, 320),
        use_flow=True,
        scenes_list_file=f"{data_root}/train_sequences_list.txt",
        with_origin_image=False,
        pipelines=pipelines),
    val=dict(
        type=dataset_type,
        root_dir=data_root,
        step=1,
        hist_len=4,
        futr_len=6,
        roi_size=(32, 32, 2),
        image_size=(1600, 900),
        resize=(512, 320),
        use_flow=True,
        scenes_list_file=f"{data_root}/val_sequences_list.txt",
        with_origin_image=False,
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
test_checkpoint = "work_dirs/checkpoints/NonARTransformerNuScenes/1.0.0/last.ckpt"
batch_size_times = 1
simple_profiler = True

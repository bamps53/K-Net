_base_ = [
    '../_base_/models/knet_s3_r50_fpn.py',
    '../common/mstrain_3x_coco_instance.py'
]

dataset_type = 'CocoDataset'
data_root = '../input/sartorius-cell-instance-segmentation-coco/'
img_norm_cfg = dict(
    mean=[128, 128, 128], std=[2.17708144, 2.17708144, 2.17708144], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal', 'vertical']),
    dict(type='RandomAffine', max_rotate_degree=360.0, max_translate_ratio=0.1,
         scaling_ratio_range=(0.75, 1.25), max_shear_degree=2.0, border=(0, 0), border_val=(128, 128, 128),
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=True,
        flip_direction=direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file="../input/sartorius-cell-instance-segmentation-coco/annotations_train.json",
        img_prefix="../input/sartorius-cell-instance-segmentation",
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file="../input/sartorius-cell-instance-segmentation-coco/annotations_val.json",
        img_prefix="../input/sartorius-cell-instance-segmentation",
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file="../input/sartorius-cell-instance-segmentation-coco/annotations_val.json",
        img_prefix="../input/sartorius-cell-instance-segmentation",
        pipeline=test_pipeline))
# we do not evaluate bbox because K-Net does not predict bounding boxes
evaluation = dict(metric=['segm'])


model = dict(
    backbone=dict(
        depth=101,
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')))

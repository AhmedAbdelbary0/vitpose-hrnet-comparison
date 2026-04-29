_base_ = ['../mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py']

codec = dict(
    type='UDPHeatmap', input_size=(288, 384), heatmap_size=(72, 96), sigma=2)

model = dict(
    backbone=dict(
        img_size=(384, 288),
    ),
    head=dict(
        decoder=codec
    )
)

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs')
]

val_dataloader = dict(
    dataset=dict(
        pipeline=val_pipeline
    )
)

test_dataloader = val_dataloader

_base_ = ['./segformer_mit-b0_8xb1-160k_light-1024x1024.py']
class_w = [1.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(
                  in_channels=[64, 128, 320, 512]
                , loss_decode=[dict(
                         type='CrossEntropyLoss',
                         use_sigmoid=False,
                         class_weight=class_w,
                         loss_weight=1.0)]))

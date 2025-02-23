_base_ = './fcn_hr18_4xb2-160k_light-512x1024.py'
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
          in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])
        , loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    )
)

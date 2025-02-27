_base_ = './fcn_hr18_4xb2-160k_light-512x1024.py'
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w18_small',
    backbone=dict(
        extra=dict(
            stage1=dict(num_blocks=(2, )),
            stage2=dict(num_blocks=(2, 2)),
            stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
            stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2)))),

    decode_head=dict(
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))

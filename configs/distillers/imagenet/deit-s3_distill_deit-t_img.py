_base_ = [
    '../../deit/deit-tiny_pt-4xb256_in1k.py'
]
# model settings
find_unused_parameters = True

# distillation settings
use_logit = True
is_vit = True

# config settings
wsld = False
dkd = False
kd = False
nkd = False
vitkd = True
flowkd = True

# method details
model = dict(
    _delete_ = True,
    type='FMKTClassificationDistiller',
    use_logit = use_logit,
    is_vit = is_vit,
    teacher_pretrained = 'https://download.openmmlab.com/mmclassification/v0/deit3/deit3-small-p16_in21k-pre_3rdparty_in1k_20221009-dcd90827.pth',
    teacher_cfg = 'configs/deit3/deit3-small-p16_64xb64_in1k.py',
    student_cfg = 'configs/deit/deit-tiny_pt-4xb256_in1k.py',
    distill_cfg = [ dict(methods=[dict(type='ViTKDLoss',
                                       name='loss_vitkd',
                                       use_this = vitkd,
                                       student_dims = 192,
                                       teacher_dims = 384,
                                       alpha_vitkd=0.00003,
                                       beta_vitkd=0.000003,
                                       lambda_vitkd=0.5,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='WSLDLoss',
                                       name='loss_wsld',
                                       use_this = wsld,
                                       temp=2.0,
                                       alpha=2.5,
                                       num_classes=1000,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='DKDLoss',
                                       name='loss_dkd',
                                       use_this = dkd,
                                       temp=1.0,
                                       alpha=1.0,
                                       beta=0.5,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='NKDLoss',
                                       name='loss_nkd',
                                       use_this = nkd,
                                       temp=1.0,
                                       gamma=1.0,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='KDLoss',
                                       name='loss_kd',
                                       use_this=kd,
                                       temp=1.0,
                                       alpha=0.5,
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='FlowLoss',
                                       name='loss_fmkd',
                                       use_this=flowkd,
                                       teacher_channel=1000,
                                       student_channel=192,
                                       loss_type="logit_based",
                                       flow_loss_type="dist",
                                       encoder_type="mlp",
                                       number=2,
                                       inference_sampling=8
                                       )
                                  ]
                         ),
                    ]
    )

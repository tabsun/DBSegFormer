_base_ = [
    '../_base_/models/ocrnet_r50-d8.py', '../_base_/datasets/mgs.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1k.py'
]
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
optimizer = dict(lr=0.01)
lr_config = dict(min_lr=2e-4)

_base_ = [
    '../_base_/models/pspnet_unet_s5-d16.py', '../_base_/datasets/mgs.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1k.py'
]
model = dict(test_cfg=dict(crop_size=(512, 512), stride=(170, 170)))
#evaluation = dict(metric='mDice')

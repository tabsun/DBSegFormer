# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005) #,
#         paramwise_cfg= dict(
#             custom_keys={
#                 'head': dict(lr_mult=.75)}))
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=2000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=50, metric='mIoU', pre_eval=True, save_best='mIoU')

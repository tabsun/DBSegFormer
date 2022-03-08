_base_ = ['./segformer_mit-b5_1024x1024_2k_mgs.py']

# model settings
model = dict(
    pretrained='pretrain/using_mit_b2.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

from .swin import BaseSwinUnet
from .restormer import BaseRestormer
from .bunet import BaseUnet

# pretrained_dict = torch.load(save_path)
# model_dict = model.state_dict()
#
# # 1. filter out unnecessary keys
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['unet.conv.weight','unet.conv.bias']}
# # 2. overwrite entries in the existing state dict
# model_dict.update(pretrained_dict)
# # 3. load the new state dict
# model.load_state_dict(pretrained_dict, strict=False)

def build_model(model, problem):
    if model == 'restormer':
        if problem == 'deraining':
            model = BaseRestormer(inp_channels=3, out_channels=3, dim=24)
        elif problem == 'denoise':
            model = BaseRestormer(inp_channels=1, out_channels=1, dim=24)
        elif problem == 'firstbreak':
            model = BaseRestormer(inp_channels=1, out_channels=2, dim=24)
        else:
            raise ValueError('Undefined problem!')
    elif model == 'swin':
        if problem == 'deraining':
            model = BaseSwinUnet(in_chans=3, num_classes=3, embed_dim=48)
        elif problem == 'denoise':
            model = BaseSwinUnet(in_chans=1, num_classes=1, embed_dim=48)
        elif problem == 'firstbreak':
            model = BaseSwinUnet(in_chans=1, num_classes=2, embed_dim=48)
        else:
            raise ValueError('Undefined problem!')
    elif model == 'unet':
        if problem == 'deraining':
            model = BaseUnet(in_channels=3, out_channels=3)
        elif problem == 'denoise':
            model = BaseUnet(in_channels=1, out_channels=1)
        elif problem == 'firstbreak':
            model = BaseUnet(in_channels=1, out_channels=2)
        else:
            raise ValueError('Undefined problem!')
    else:
        raise ValueError('Undefined model!')
    return model
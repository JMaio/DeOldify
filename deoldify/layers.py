from fastai.layers import *
from fastai.torch_core import *
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn import GroupNorm

# override NormTypes
normtypes = [m.name for m in NormType] + ['LayerNorm', 'InstanceNorm', 'GroupNorm']
NormType = Enum('NormType', normtypes)

# The code below is meant to be merged into fastaiv1 ideally


def custom_conv_layer(
    ni: int,
    nf: int,
    ks: int = 3,
    stride: int = 1,
    padding: int = None,
    bias: bool = None,
    is_1d: bool = False,
    norm_type: Optional[NormType] = NormType.Batch,
    use_activ: bool = True,
    leaky: float = None,
    transpose: bool = False,
    init: Callable = nn.init.kaiming_normal_,
    self_attention: bool = False,
    extra_bn: bool = False,
):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero) or extra_bn == True
    gn = norm_type == NormType.GroupNorm
    if bias is None:
        bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = init_default(
        conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding),
        init,
    )

    # spectral norm by default
    conv = spectral_norm(conv)

    layers = [conv]
    if use_activ:
        layers.append(relu(True, leaky=leaky))

    # elif norm_type == NormType.GroupNorm:
    #     # https://pytorch.org/docs/stable/nn.html#groupnorm
    #     # >>> input = torch.randn(20, 6, 10, 10)
    #     # >>> # Separate 6 channels into 3 groups
    #     # >>> m = nn.GroupNorm(3, 6)
    #     # >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
    #     # >>> m = nn.GroupNorm(6, 6)
    #     # >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
    #     # >>> m = nn.GroupNorm(1, 6)
    #     # >>> # Activating the module
    #     # >>> output = m(input)
        
    #     # "We set G = 32 for GN by default."" (Wu, He 2018 - Group Norm paper)
    #     # torch.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True)
    #     group_norm = GroupNorm(32, 3)
    #     conv = group_norm(conv)

    if norm_type == NormType.Batch:
        layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    elif norm_type == NormType.InstanceNorm:
        layers.append(GroupNorm(nf, nf))
    elif norm_type == NormType.LayerNorm:
        layers.append(GroupNorm(1, nf))
    elif norm_type == NormType.GroupNorm:
        # if channels is divisible by 32, split into
        # 32 groups, otherwise, use simple grouping
        gs = 32 if nf % 32 == 0 else 1
        layers.append(GroupNorm(gs, nf))
    
    if self_attention:
        layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)

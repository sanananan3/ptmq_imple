import torch.nn as nn
import torch
from .resnet import BasicBlock, Bottleneck, resnet18, resnet50  # noqa: F401
# from .regnet import ResBottleneckBlock, regnetx_600m, regnetx_3200m  # noqa: F401
# from .mobilenetv2 import InvertedResidual, mobilenetv2  # noqa: F401
# from .mnasnet import _InvertedResidual, mnasnet  # noqa: F401
from quantize.quantized_module import QuantizedLayer, QuantizedBlock, Quantizer   # noqa: F401

 # 우선은 다른 모델말고 resnet 부터.. 
 # QuantBasicBlock , QuantBottleneck : resnet 에서 쓰이는 기본 블록 

 
class QuantBasicBlock(QuantizedBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """
    def __init__(self, org_module: BasicBlock, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.conv1_relu = QuantizedLayer(org_module.conv1, org_module.relu1, w_qconfig, a_qconfig)
        self.conv2 = QuantizedLayer(org_module.conv2, None, w_qconfig, a_qconfig, qoutput=False)
        if org_module.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantizedLayer(org_module.downsample[0], None, w_qconfig, a_qconfig, qoutput=False)
        self.activation = org_module.relu2
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1_relu(x)
        out = self.conv2(out)
        out += residual
        out = self.activation(out)
        if self.qoutput:
            out = self.block_post_act_fake_quantize(out)
        return out


class QuantBottleneck(QuantizedBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
    """
    def __init__(self, org_module: Bottleneck, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.conv1_relu = QuantizedLayer(org_module.conv1, org_module.relu1, w_qconfig, a_qconfig)
        self.conv2_relu = QuantizedLayer(org_module.conv2, org_module.relu2, w_qconfig, a_qconfig)
        self.conv3 = QuantizedLayer(org_module.conv3, None, w_qconfig, a_qconfig, qoutput=False)

        if org_module.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantizedLayer(org_module.downsample[0], None, w_qconfig, a_qconfig, qoutput=False)
        self.activation = org_module.relu3
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1_relu(x)
        out = self.conv2_relu(out)
        out = self.conv3(out)
        out += residual
        out = self.activation(out)
        if self.qoutput:
            out = self.block_post_act_fake_quantize(out)
        return out


specials = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck
}


def load_model(config):
    config['kwargs'] = config.get('kwargs', dict())
    model = eval(config['type'])(**config['kwargs'])
    checkpoint = torch.load(config.path, map_location='cpu')
    if config.type == 'mobilenetv2':
        checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint)
    return model

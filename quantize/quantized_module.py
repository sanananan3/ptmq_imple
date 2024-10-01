from torch import nn
from .observer import MSEFastObserver, MinMaxObserver, AvgMinMaxObserver, MSEObserver, \
    AvgMSEObserver, AvgMSEFastObserver
from .quant import AdaRoundFakeQuantize, FixedFakeQuantize, LSQFakeQuantize, LSQPlusFakeQuantize
import torch.nn.functional as F

# 양자화된 레이어 (conv2d, linear, embedding) 및 모듈을 구축 

ObserverDict = {
    'MinMaxObserver':           MinMaxObserver,                                    # noqa: E241
    'AvgMinMaxObserver':        AvgMinMaxObserver,                                 # noqa: E241
    'MSEObserver':              MSEObserver,                                       # noqa: E241
    'AvgMSEObserver':           AvgMSEObserver,                                    # noqa: E241
    'MSEFastObserver':          MSEFastObserver,                                   # noqa: E241
    'AvgMSEFastObserver':       AvgMSEFastObserver,                                # noqa: E241
}

FakeQuantizeDict = {
    'FixedFakeQuantize':     FixedFakeQuantize,                                    # noqa: E241
    'LSQFakeQuantize':       LSQFakeQuantize,                                      # noqa: E241
    'LSQPlusFakeQuantize':   LSQPlusFakeQuantize,                                  # noqa: E241
    'AdaRoundFakeQuantize':  AdaRoundFakeQuantize,                                 # noqa: E241
}


# 1. ActivationQuantizer => activation을 quantize 하는 FakeQuantize 객체를 생성 (quantized된 activation은 주로 레이어의 출력 부분에서 사용됨)
def ActivationQuantizer(a_qconfig):
    return FakeQuantizeDict[a_qconfig.quantizer](ObserverDict[a_qconfig.observer], bit=a_qconfig.bit,
                                                 symmetric=a_qconfig.symmetric, ch_axis=a_qconfig.ch_axis)

# 2. WeightQuantizer => weight를 quantize 하는 FakeQuantize 객체를 생성 (quantized된 activation은 주로 레이어의 출력 부분에서 사용됨)

def WeightQuantizer(w_qconfig):
    return FakeQuantizeDict[w_qconfig.quantizer](
            ObserverDict[w_qconfig.observer],
            bit=w_qconfig.bit,
            symmetric=w_qconfig.symmetric,
            ch_axis=w_qconfig.ch_axis)


class QuantizedOperator():
    pass


class QConv2d(QuantizedOperator, nn.Conv2d):

    # 양자화된 Conv2d 레이어 정의 

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 groups,
                 bias,
                 padding_mode,
                 w_qconfig):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input): # 가중치를 양자화한 후 conv2d의 기본 연산 수행 
        print(f"QConvd input shape: {input.shape}")
        print(f"Expected weight shape: {self.weight.shape}")
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)


class QLinear(QuantizedOperator, nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias,
                 w_qconfig):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input): # weight quantizaton한 후 linear 연산에 사용 
        return F.linear(input, self.weight_fake_quant(self.weight), self.bias)


class QEmbedding(QuantizedOperator, nn.Embedding):

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx,
                 max_norm,
                 norm_type,
                 scale_grad_by_freq,
                 sparse,
                 _weight,
                 w_qconfig):
        super().__init__(num_embeddings=num_embeddings,
                         embedding_dim=embedding_dim,
                         padding_idx=padding_idx,
                         max_norm=max_norm,
                         norm_type=norm_type,
                         scale_grad_by_freq=scale_grad_by_freq,
                         sparse=sparse,
                         _weight=_weight)
        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input):
        return F.embedding( # 가중치 양자화한 후 f.embedding 연산 수행 
            input, self.weight_fake_quant(self.weight), self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


module_type_to_quant_weight = { # 기본 레이어 타입을 양자화된 레이어로 매핑하는 dict 
    nn.Linear: QLinear,
    nn.Conv2d: QConv2d,
    nn.Embedding: QEmbedding,
}


def get_module_args(module): # 레이어에서 필요 인자를 추출한다. 해당 레이어의 인자를 dict 형태로 반환한다. 
    if isinstance(module, nn.Linear):
        return dict(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None
            )
    elif isinstance(module, nn.Conv2d):
        return dict(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            )
    elif isinstance(module, nn.Embedding):
        return dict(
            num_embeddings=module.num_embeddings,
            embedding_dim=module.embedding_dim,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
            _weight=None,
        )
    else:
        raise NotImplementedError


def Quantizer(module, config): # 모듈을 양자화하는 함수 
    if module is None: # 모듈이 없을 경우, activation을 양자화한다. 
        return ActivationQuantizer(a_qconfig=config)
    module_type = type(module)
    if module_type in module_type_to_quant_weight: #있으면 weight 도 양자화 
        kwargs = get_module_args(module) # 모듈의 인자 가져오고 
        qmodule = module_type_to_quant_weight[module_type](**kwargs, w_qconfig=config)
        qmodule.weight.data = module.weight.data.clone()
        if getattr(module, 'bias', None) is not None:
            qmodule.bias.data = module.bias.data.clone()
        return qmodule # 양자화된 레이어 생성 

    return module


class QuantizedModule(nn.Module): # 양자화된 모듈의 상위 클래스 
    def __init__(self):
        super().__init__()


class QuantizedLayer(QuantizedModule): # 단일 레이어를 양자화하는 클래스 
    def __init__(self, module, activation, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.module = Quantizer(module, w_qconfig) # 주어진 모듈을 양자화하고 
        # activation과 출력(post_activation)을 양자화한다. 
        self.activation = activation 
        if qoutput:
            self.layer_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x):
        x = self.module(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.qoutput:
            x = self.layer_post_act_fake_quantize(x)
        return x
    
    def set_activation_quantization_bit(self, bit):
        "activation quantization bit 수 설정하기"
        if hasattr(self, 'layer_post_act_fake_quantize'):
            self.layer_post_act_fake_quantize.set_bit(bit)


class QuantizedBlock(QuantizedModule): # 여러 레이어를 양자화하는 블록 클래스 
    def __init__(self):
        super().__init__()

import torch
import torch.nn as nn  # Pytorch의 신경망 모듈을 상속받기 위한 기본 클래스 nn.module
from .observer import ObserverBase # 양자화를 위한 통계 수집 => Observer가 수행 (입력 텐서의 최소-최대값 관찰.. 양자화 범위 계산 등에 이용)


# fake_quant => util_quant 에서 정의한 기본 연산 함수들을 모듈화해서 사용한다. 
# 양자화를 다양한 방식으로 구현한 다양한 클래스들이 있음 

# 고정된 양자화 (FixedFakeQuantize - fixed-precision 아마... ), 학습 가능한 양자화 (LSQ, LSQPlus - mixed-precision 아마), AdaRound 등 다양한 양자화 방식 정의 
# 실제 모델에 적용할 수 있는 양자화 모듈을 정의하고, 양자화된 모델을 통해 추론할 수 있도록 한다. 


from .util_quant import ( # 양자화 함수들 import (텐서 및 채널 별 양자화 방식 등)
    fake_quantize_per_channel_affine, # 채널 별 양자화 수행
    fake_quantize_per_tensor_affine, # 텐서 전체를 양자화
    fake_quantize_learnable_per_tensor_affine_training, # 텐서 단위에서 학습 가능한 양자화
    fake_quantize_learnable_per_channel_affine_training, # 채널 단위에서 학습 가능한 양자화
    fake_quantize_learnableplus_per_channel_affine_training, # 채널 별로 더 세밀하게 학습 가능한 양자화
    fake_quantize_learnableplus_per_tensor_affine_training, # 텐서 단위로 더 세밀하게 학습 가능 
)


class QuantizeBase(nn.Module): # 양자화 기본 기능 제공하는 클래스 

    def __init__(self, observer=ObserverBase, bit=8, symmetric=False, ch_axis=-1):
        super().__init__() # nn.Module 초기화 
        self.observer = observer(bit=bit, symmetric=symmetric, ch_axis=ch_axis) # observer: 입력 텐서의 최소/최대값 관찰 => 양자화 범위 설정한다. 
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis # 채널 단위로 양자화 할 경우 채널의 축 지정
        self.observer_enabled = 0
        self.fake_quant_enabled = 0
        self.quant_min = self.observer.quant_min # 양자화 최소값 
        self.quant_max = self.observer.quant_max # 양자화 최대값 
        self.drop_prob = 1.0 # dropout 확률 

    def set_bit(self, bit): # 양자화 bit 수 설정하기 
        self.observer.set_bit(bit)
        self.bit = bit
        self.quant_min = self.observer.quant_min
        self.quant_max = self.observer.quant_max

    def set_name(self, name): # 양자화 모듈에 이름 지정해주기 
        self.name = name

    @torch.jit.export
    def calculate_qparams(self): # scale factor 및 zero-point 계산 
        return self.observer.calculate_qparams()

    @torch.jit.export
    def disable_observer(self):
        self.observer_enabled = 0 # observer 비활

    @torch.jit.export
    def enable_observer(self):
        self.observer_enabled = 1 # observer 활성화 

    @torch.jit.export
    def disable_fake_quant(self):
        self.fake_quant_enabled = 0 # fake_quant 비활

    @torch.jit.export
    def enable_fake_quant(self):
        self.fake_quant_enabled = 1 # fake_quant 활성화

    @torch.jit.export
    def extra_repr(self): # 디버깅이나 로그에 사용될 수 있는 문자열 정의 

        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'symmetric={}, bit={}, ch_axis={}, quant_min={}, quant_max={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.symmetric, self.bit, self.ch_axis,
                   self.quant_min, self.quant_max)


        # save_to_state_Dict => 학습된 모델 저장할 때 scale이랑 zero-point 도 함께 저장한다. 

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                if name == 'scale':
                    if isinstance(self.scale, nn.Parameter):
                        self.scale.data = torch.ones_like(val.to(self.scale.device))
                    else:
                        self.scale.resize_(val.shape)
                else:
                    assert name == 'zero_point'
                    if isinstance(self.zero_point, nn.Parameter):
                        self.zero_point.data = torch.ones_like(val.to(self.zero_point.device))
                    else:
                        self.zero_point.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == 'scale':
                        self.scale.copy_(val)
                    else:
                        assert name == 'zero_point'
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class FixedFakeQuantize(QuantizeBase): # 고정된 양자화 적용 . 채널 또는 텐서 단위에서 fake quantization을 수행한다. 

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float)) # scale 과 zero point 값을 모델에 저장하지만 학습하지는 않는 변수로써 등록한다. 
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.drop_prob = 1.0 # 기본 설정 값 1.0 => 모든 값이 양자화된다. 
        # drop_prob는 Fake Quantization에서 일부 값을 양자화하지 않고 원래 값을 유지하는것이다. => qdrop 이어서 그런것 같음..?? 
        # drop_prob 확률적 양자화를 수행한다. drop_prob가 1보다 작은 경우에 입력 텐서의 아무 값을 원래 값으로 유지한다. 
        # drop_prob이 1이면 모두 다 양자화하는 것이다. 



    def forward(self, X):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled == 1:
            if self.drop_prob < 1.0: 
                x_orig = X # quantization 하지 않고 원본 데이터로 저장 

            if self.ch_axis != -1:
                X = fake_quantize_per_channel_affine(
                    X, self.scale.data, self.zero_point.data.int(), self.ch_axis,
                    self.quant_min, self.quant_max)
            else:
                X = fake_quantize_per_tensor_affine(
                    X, self.scale.item(), self.zero_point.item(),
                    self.quant_min, self.quant_max)
            if self.drop_prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.drop_prob, X, x_orig) # 확률적으로 일부 값을 유지 
                return x_prob # 확률적으로 선택된 값을 반환 

        return X


class LSQFakeQuantize(QuantizeBase): # 학습 가능한 양자화 (Learned Step Size Quantization)을 구현. 스케일을 학습 할 수 있도록 한다. 

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, use_grad_scaling=True):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        
        '''scale을 학습 가능한 파라미터로 설정하여 학습 중에 스케일을 조정할 수 있다.'''
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float))

        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        self.use_grad_scaling = use_grad_scaling
        self.drop_prob = 1.0 

    def forward(self, X):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.ch_axis != -1:
                self.scale.data = torch.ones_like(_scale)
                self.zero_point.resize_(_zero_point.shape)

            self.scale.data.copy_(_scale)
            self.zero_point.copy_(_zero_point)
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())

        if self.fake_quant_enabled == 1:
            if self.drop_prob < 1.0:
                x_orig = X
            if self.ch_axis != -1:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnable_per_channel_affine_training(
                    X, self.scale, self.zero_point.data.int(), self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnable_per_tensor_affine_training(
                    X, self.scale, self.zero_point.item(), self.quant_min, self.quant_max, grad_factor)
            if self.drop_prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.drop_prob, X, x_orig)
                return x_prob
        return X


class LSQPlusFakeQuantize(QuantizeBase): 

    # 여기에서는 scale + zero point 도 학습 가능한 파라미터로 추가해준다. 
    # Mixed Precision 아님!!!!! scale factor랑 zero point 만 학습 가능한거지, bit-width가 레이어마다 바뀌는 건 아님 . 
    # 논문에서는 편리성을 위해서 zero point 도 learnable 하게 하지는 않았고, scale factor 만 고려했음 

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, use_grad_scaling=True):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float))
        self.zero_point = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float)) # 이부분!!!
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        self.use_grad_scaling = use_grad_scaling
        self.drop_prob = 1.0 # 모든 값이 다 양자화됨 

    def forward(self, X):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.ch_axis != -1:
                self.scale.data = torch.ones_like(_scale)
                self.zero_point.data = torch.zeros_like(_zero_point.float())
            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point.float())
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())
            self.zero_point.data.clamp_(self.quant_min, self.quant_max).float()

        if self.fake_quant_enabled == 1:
            if self.drop_prob < 1.0:
                x_orig = X
            if self.ch_axis != -1:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnableplus_per_channel_affine_training(
                    X, self.scale, self.zero_point, self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnableplus_per_tensor_affine_training(
                    X, self.scale, self.zero_point, self.quant_min, self.quant_max, grad_factor)
            if self.drop_prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.drop_prob, X, x_orig)
                return x_prob
        return X


class AdaRoundFakeQuantize(QuantizeBase):

    # adaround 양자화 방식 구현 ... 가중치를 더욱 세밀하게 조정하여 양자화 오차를 줄인다. 
    """
    self.adaround=True: turn on up or down forward
    self.adaround=False: turn on round-to-nearest forward
    based on the FixedFakeQuantize
    """

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)

        # 여기에서는 scale 이랑 zero_point 가 학습되진 않지만 모델의 상태로 저장되는 변수임 
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.adaround = False  # adaround 활성화 여부 => adaround: 가중치의 값이 중간값에 가까운 경우 업 또는 다운할지 학습을 통해 결정한다. 
        self.gamma, self.zeta = -0.1, 1.1 # hard sigmoid 함수의 하한과 상한을 결정하는 값으로, 양자화 시 중간값이 업 또는 다운되는 확률을 조정하는데 사용한다. 

    def init(self, weight_tensor: torch.Tensor, round_mode):
        self.adaround = True # adaround 모드 활성화하기 
        self.round_mode = round_mode # rounding 모드 설정하기 
        self.init_alpha(x=weight_tensor.data.clone().detach()) # alpha 값 초기화 (양자화 과정에서 값이 업 또는 다운될 지 결정하는데 사용)

    def init_alpha(self, x: torch.Tensor):

        # alpha 파라미터는 양자화할 때 중간값이 업 될지 다운 될 지를 결정한다.
        # hard sigmoid 함수를 통하여 학습되며, 양자화 경계에서의 조정이 가능해진다. 
        if self.ch_axis != -1:
            new_shape = [1] * len(x.shape)
            new_shape[self.ch_axis] = x.shape[self.ch_axis]
            scale = self.scale.data.reshape(new_shape)
        else:
            scale = self.scale.data
        x_floor = torch.floor(x / scale)
        if self.round_mode == 'learned_hard_sigmoid':
            rest = (x / scale) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = torch.nn.Parameter(alpha)
        else:
            raise NotImplementedError

    def rectified_sigmoid(self):
        """generate rounding mask.
            학습된 alpha 값에 따라 양자화 값이 업 또는 다운되는 확률을 결정하는 mask 를 생성한다. 
        """
        return ((self.zeta - self.gamma) * torch.sigmoid(self.alpha) + self.gamma).clamp(0, 1) 

    def adaround_forward(self, X, hard_value=False):
        if self.ch_axis != -1:
            new_shape = [1] * len(X.shape)
            new_shape[self.ch_axis] = X.shape[self.ch_axis]
            scale = self.scale.data.reshape(new_shape)
            zero_point = self.zero_point.data.int().reshape(new_shape)
        else:
            scale = self.scale.item()
            zero_point = self.zero_point.item()
        X = torch.floor(X / scale) # 양자화된 값의 정수 부분

        if hard_value: # hard_value 가 참이면 학습된 alpha 가 0보다 크거나 같을 때만 반올림한다... 
            X += (self.alpha >= 0).float()  # 알파값에 따라 양자화 업/다운 결정
        else:
            X += self.rectified_sigmoid() # hard value 가 false 로 설정되어있을 경우에 hard sigmoid로 계산된 확률을 기반으로 결정
        X += zero_point # zero point 적용
        X = torch.clamp(X, self.quant_min, self.quant_max) # 최소/최대값 제한
        X = (X - zero_point) * scale # scale 적용 

        return X

    def get_hard_value(self, X): 
        # hard_value : 양자화 과정에서 가중치 값을 강제적으로 (확률적이 아님.. 무조건 반올림 하거나 내림하거나 둘중하나임) 반올림할지 결정하는 파라미터
        X = self.adaround_forward(X, hard_value=True)
        return X

    def forward(self, X):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled == 1:
            if not self.adaround:
                if self.ch_axis != -1:
                    X = fake_quantize_per_channel_affine(
                        X, self.scale.data, self.zero_point.data.int(), self.ch_axis,
                        self.quant_min, self.quant_max)
                else:
                    X = fake_quantize_per_tensor_affine(
                        X, self.scale.item(), self.zero_point.item(),
                        self.quant_min, self.quant_max)
            else:
                if not hasattr(self, 'alpha'):
                    raise NotImplementedError
                if self.round_mode == 'learned_hard_sigmoid':
                    X = self.adaround_forward(X)
                else:
                    raise NotImplementedError
        return X

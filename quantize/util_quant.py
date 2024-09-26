import torch


# util_quant : 유틸리티 함수를 제공. 양자화를 도울 여러 함수가 정의되어있음 
# 양자화를 수행하는 기본적인 수학적 연산을 정의 
# fake quantization을 하기 위한 기초적인 연산을 수행하는 것 . 

def round_ste(x: torch.Tensor): # 

    """
    Straight-Through Esimator 방식으로 rounding을 수행 
    양자화 과정에서 반올림 연산이 미분 불가능하므로 학습 중에는 연산을 미분 가능한 형태로 처리하지만 
    back propagation 과정에서는 미분을 무시하게끔 처리  
        """

def fake_quantize_per_tensor_affine(x, scale, zero_point, quant_min, quant_max):
    x_int = round_ste(x / scale) + zero_point # x/scale (정규화) => round_ste (반올림하여 정수값에 가깝게 만듬) => zero point 를 더하기 
    x_quant = torch.clamp(x_int, quant_min, quant_max) # clamp => 고정 / 양자화된 값 x_int 를 지정된 범위 min-max로 클리핑하여 범위 밖의 값들을 제거 
    x_dequant = (x_quant - zero_point) * scale # 다시 실수 범위로 복원 => 이를 통해 텐서를 quantizaiton-reconsturction 과정에 통과시킴 
    return x_dequant


def fake_quantize_per_channel_affine(x, scale, zero_point, ch_axis, quant_min, quant_max): # 채널 별로 양자화 수행
    new_shape = [1] * len(x.shape) # 입력 텐서 x의 차원 수와 동일한 길이의 리스트 생성 , 각 요소를 1로 채운다. 
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = scale.reshape(new_shape) # scale을 해당 채널 차원에 맞게 reshape 하여 적용한다. 
    zero_point = zero_point.reshape(new_shape) # zero point 도 마찬가지 
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnable_per_tensor_affine_training(x, scale, zero_point, quant_min, quant_max, grad_factor):
    # 학습 가능한 양자화를 수행 
    scale = grad_scale(scale, grad_factor) # scale에 gradient scaling 을 적용한다. 학습 중에 스케일이 조정될 수 있음 
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnable_per_channel_affine_training(x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape) # 채널 단위로 scale을 학습 가능한 양자화를 수행 
    zero_point = zero_point.reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnableplus_per_tensor_affine_training(x, scale, zero_point, quant_min, quant_max, grad_factor):
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    scale = grad_scale(scale, grad_factor)
    zero_point = grad_scale(zero_point, grad_factor) # 채널 단위로 zero point도 학습 가능한 양자화를 수행 
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnableplus_per_channel_affine_training(x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = grad_scale(zero_point, grad_factor).reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def grad_scale(t, scale):  # gradient scaling 을 수행하는 함수 
    return (t - (t * scale)).detach() + (t * scale) # 텐서 t에서 스케일링된 값을 뺀 결과 

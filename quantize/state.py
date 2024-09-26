import logging
from .quant import QuantizeBase
logger = logging.getLogger("quantization")

# state.py => quantization 모델의 상태 (calibration 모드, quantization 모드, disable 모드...)를 관리하는 기능 제공 



# 1. enable_calibration_woquantization => without quantization, calibraiton 모드에서 ... 관찰자 Observer만 활성화한다. 양자화 없이 보정(calib)만 수행

def enable_calibration_woquantization(model, quantizer_type='fake_quant'):

    logger.info('Enable observer and Disable quantize for {}'.format(quantizer_type))
    for name, submodule in model.named_modules(): # 모델의 모든 서브모듈 순회 => 양자화 레이어 탐색하기 
        if isinstance(submodule, QuantizeBase): # 양자화 레이어라면
            if quantizer_type not in name: # quantizer_type 에 포함되어있지 않은 서브모듈 이름이라면 .... 
                logger.debug('The except_quantizer is {}'.format(name))
                submodule.disable_observer() # observer도 비활성화 
                submodule.disable_fake_quant() # fake 양자화도 비활성화 
                continue
            # quantizer_type에 포함되어있다면 
            logger.debug('Enable observer and Disable quant: {}'.format(name))
            submodule.enable_observer() # observer 만 활성화 
            submodule.disable_fake_quant()


def enable_quantization(model, quantizer_type='fake_quant'):
    logger.info('Disable observer and Enable quantize.') # 관찰자 비활 양자화 활성화 
    for name, submodule in model.named_modules(): 
        if isinstance(submodule, QuantizeBase): # 양자화 레이어인지 확인하기 .. quantizebase에서 파생된 레이어들만 대상
            if quantizer_type not in name:
                logger.debug('The except_quantizer is {}'.format(name))
                submodule.disable_observer() # quantizer_type 에 없다면 다 비활 
                submodule.disable_fake_quant()
                continue
            logger.debug('Disable observer and Enable quant: {}'.format(name))
            submodule.disable_observer()
            submodule.enable_fake_quant() # observer 비활, 양자화 활성화 


def disable_all(model): # observer + quantize 모두 비활 
    logger.info('Disable observer and disable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            logger.debug('Disable observer and disable quant: {}'.format(name))
            submodule.disable_observer()
            submodule.disable_fake_quant()

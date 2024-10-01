import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_path, '..')

# sys.path에 quantize 디렉토리 경로 추가하기


sys.path.append(project_root)

import numpy as np  # noqa: F401
import copy
import time
import torch
import torch.nn as nn # 위에꺼 모두 다양한 연산 및 데이터 조작을 위한 라이브러리들 
import logging # 로그 출력을 위한 라이브러리 
import argparse # 커맨드라인 인자를 처리하기 위한 라이브러리 
import imagenet_utils # imagenet_utils.py 를 import 해왔다. 이건 imagenet 데이터셋 로드 및 처리와 관련된 함수들을 모아둔 유틸리티임 
from reconstruct import reconstruction # mfm 과 gd-loss를 고려하지 않은 함수 
from reconstruct import reconstruction_with_mfm_gd_loss # mfm 과 gd-loss 고려한 함수 
from fold_bn import search_fold_and_remove_bn, StraightThrough # 배치 norm 레이어를 접거나 패스하는 기능 처리 
from optimize.imagenet_utils import parse_config

'''
batch norm 을 접거나 패스 하는 이유?? 
: BN 을 fold 한다는 의미 => BN 레이어의 효과를 이전 레이어 (보통 Conv)에 병합하는 것이다. 이를 통하여 추가적인 BN 연산을 피하고 모델을 간소화할 수 있음 
 why? BN은 학습 중에는 중요한 역할을 하지만, 추론 시에는 사실 꼭 필요 X ... BN을 Conv2d에 병합을 하면 추론 시간에 연산량을 줄일 수 잇음 
: BN을 패스한다. -> StraightThrough 를 통해서... 모델에서 BN 레이어를 없애고, 아무일도 하지 않는 레이어 (StraightThrough) 레이어를 배치하는거임~~ 
입력을 그대로 반환하는 레이어임.. 

'''
from model import load_model, specials # 모델 로딩하고 special (특수한 양자화 레이어 관련) 정의 
from quantize.state import enable_calibration_woquantization, enable_quantization, disable_all
from quantize.quantized_module import QuantizedLayer, QuantizedBlock
from quantize.quant import QuantizeBase
from quantize.observer import ObserverBase # 양자화 관련 함수들 import 


logger = logging.getLogger('qdrop') # logger 객체 생성 (프로그램 내에서 로그 메시지 기록용) - gdrop 이름의 로거 인스턴스 생성 
logging.basicConfig(level=logging.INFO, format='%(message)s') # INFO 레벨의 메시지 출력 형식 지정해주기 ... 


# 여기에 main 있음. ... ImageNet 데이터셋으로 실행하는 메인 스크립트 


# 1. quantize_model => 모델을 양자화하는 함수 : 양자화 설정을 기반으로 모델의 레이어를 양자화된 레이어로 교체한다. 

def quantize_model(model, config_quant): 

    # replace_module => 모델의 레이어를 양자화된 버전으로 교체하는 재귀적 함수 
    # 모듈을 순회하며 각 레이어를 양자화된 버전으로 교체한다. 


# ====================================================================================================

    def replace_module(module, w_qconfig, a_qconfig, qoutput=True):
        ''' module = nn.Conv2d, nn.Linear 등.... nn.Module을 상속받아 구현된 레이어가 모듈로 취급'''
        childs = list(iter(module.named_children())) # 모듈의 자식 레이어들을 가져와서 childs 리스트에 저장한다. Conv2D, Linear 레이어들의 각각의 레이어 
        st, ed = 0, len(childs) # 순회의 시작과 끝 정의 
        prev_quantmodule = None # 이전에 양자화된 모듈을 저장한다. 

        while(st < ed):
            ''' 마지막 레이어(출력 레이어)는 양자화를 적용 안한다... qoutput=False 
            왜??? 양자화 적용하면 출력값의 정확도가 손상되니까..! '''
            tmp_qoutput = qoutput if st == ed - 1 else True # 마지막 자식 모듈에 대해서는 qoutput 설정하고 (마지막 레이어만 양자화 x ) 나머지 else => 걍 true 로 지정 
            name, child_module = childs[st][0], childs[st][1] # name : 모듈 이름, child_module : 해당 모듈 

            if type(child_module) in specials: # child_module이 specials dictionary 에 있는 타입이라면 그 레이어를 특수한 양자화 블록으로 교체한다. 
                setattr(module, name, specials[type(child_module)](child_module, w_qconfig, a_qconfig, tmp_qoutput))

            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantizedLayer(child_module, None, w_qconfig, a_qconfig, qoutput=tmp_qoutput)) # 양자화된 레이어로 교체 
                prev_quantmodule = getattr(module, name) # 현재 양자화된 모듈을 저장 

            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)): # activation function이 있을 때 
                if prev_quantmodule is not None:
                    prev_quantmodule.activation = child_module # 이전에 양자화된 모듈에 연결해주고
                    setattr(module, name, StraightThrough())  # relu를 straightthrough로 대체해준다. 
                else:
                    pass

            elif isinstance(child_module, StraightThrough): # 이미 straightThrough 인 경우에는 그대로 유지 (이미 bn이 x )
                pass

            else: #  자식 모듈의 하위모듈이  더 있으면 재귀적으로 replace_module 호출 => 계속 bn을 conv로 합쳐주기 .. 남은 거 다!!!
                replace_module(child_module, w_qconfig, a_qconfig, tmp_qoutput)
            st += 1  # 재귀호출 끝나고 그 다음 형제 모듈로 넘어가기 

# =============================================================================================================

    replace_module(model, config_quant.w_qconfig, config_quant.a_qconfig, qoutput=False) # replace ~ 함수 호출해서 모델의 레이어 양자화하기 
    w_list, a_list = [], [] # 양자화된 weight 레이어 리스트, activation 레이어 리스트 

    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase) and 'weight' in name:
            w_list.append(module)
        if isinstance(module, QuantizeBase) and 'act' in name:
            a_list.append(module)
    w_list[0].set_bit(8)
    w_list[-1].set_bit(8) # 첫번째-마지막 가중치 레이어는 8비트로 양자화 

    'the image input has already been in 256, set the last layer\'s input to 8-bit'
    a_list[-1].set_bit(8) # 마지막 활성화 레이어도 8비트로 양자화 

    logger.info('finish quantize model:\n{}'.format(str(model))) 
    return model # 양자화가 완료된 모델을 반환 




# 2. get_cali_data => 학습 데이터셋에서 일부를 선택하여 quantization parameter 조정에 사용한다. 

def get_cali_data(train_loader, num_samples):
    cali_data = []
    for batch in train_loader:
        cali_data.append(batch[0]) # 학습 데이터셋에서 ( 입력 데이터 == batch[0] ) 만을 모아서 calibration 데이터로 저장한다. 
        if len(cali_data) * batch[0].size(0) >= num_samples: # len(cali_data) = 몇 개의 배치가 잇는ㄴ지, batch(0).size(0) = 하나의 배치에 몇 개의 샘플이 있는지 ... 즉 , 배치 수 x 배치 크기 = 총 샘플의 갯수 
            break
    return torch.cat(cali_data, dim=0)[:num_samples] # 데이터를 하나의 텐서로 병합하여 반환한다. 


def main(config_path):

    
   #  device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = imagenet_utils.parse_config(config_path) # config.yaml 설정 파일을 읽어오고, 랜덤 시드 정하기 
    imagenet_utils.set_seed(config.process.seed)

    'cali data'
    train_loader, val_loader = imagenet_utils.load_data(**config.data) # 데이터셋 로드하고
    cali_data = get_cali_data(train_loader, config.quant.calibrate) # 학습 데이터에서 calibration 데이터 가져오기 

    'model'
    model = load_model(config.model) # 모델 로드하고 
    search_fold_and_remove_bn(model) # BN 찾아서 fold 

    if hasattr(config, 'quant'): # 양자화 설정이 있다면....
        model = quantize_model(model, config.quant) # 모델을 양자화 하기 


    # model = model.to(device)
    model.cuda()
    model.eval() # 모델 gpu에 올리고 평가 모드로 전환하기 
    fp_model = copy.deepcopy(model) # fp_model에 원본 모델 복사해서 
    disable_all(fp_model) # 양자화 되지 않은 상태 유지하기 

    for name, module in model.named_modules():
        if isinstance(module, ObserverBase): # 모든 양자화된 레이어에 대해서 관찰자 설정하기 
            module.set_name(name) # 양자화 레이어의 각 모듈에 이름을 할당하여 관리를 쉽게 한다. 

    # calibrate first
    with torch.no_grad():

        '양자화 된 모델에 대해서 activation 및 weight quantization 수행, calibration 데이터 이용해서 양자화 parameter 조정'
        st = time.time()
        enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
        model(cali_data[: 256].cuda())
        enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
        model(cali_data[: 2].cuda())
        ed = time.time()
        logger.info('the calibration time is {}'.format(ed - st))

    if hasattr(config.quant, 'recon'): # reconstruction 설정 있으면
        enable_quantization(model) # 양자화 활성화하고 재구성 들어가기
 
        def recon_model(module: nn.Module, fp_module: nn.Module):
            """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
            """
            for name, child_module in module.named_children():
                if isinstance(child_module, (QuantizedLayer, QuantizedBlock)):
                    logger.info('begin reconstruction for module:\n{}'.format(str(child_module)))
                    # 각 block 별로 reconstruction 적용하기 
                    # reconstruction(model, fp_model, child_module, getattr(fp_module, name), cali_data, config.quant.recon)
                    reconstruction_with_mfm_gd_loss(model, fp_model, child_module, getattr(fp_module, name), cali_data, config)
                else:
                    recon_model(child_module, getattr(fp_module, name))
        # Start reconstruction
        recon_model(model, fp_model)

    enable_quantization(model) # reconsturction 모델을 activate 하기 
    imagenet_utils.validate_model(val_loader, model) # validation 데이터 셋 이용해서 성능 평가하기 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='configuration',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()

    config_path = os.path.join(project_root,'exp','w8a8','r18','config.yaml')

    main(config_path)

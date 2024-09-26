
import sys
import os

# 현재 경로에 optimize 디렉토리가 있는지 확인하고 추가
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, 'optimize'))


import torch
from optimize.reconstruct import reconstruction_with_mfm_gd_loss
from model import load_model
from optimize.imagenet_utils import load_data, parse_config
from optimize.main import quantize_model
from quantize.quantized_module import QuantizedLayer, QuantizedBlock


def test_module_value():
    # config 파일을 불러옵니다. 이 경로는 reconstruct.py에서 사용하는 것과 동일하게 설정해야 합니다.
    config_path = os.path.join(current_path, 'exp','w8a8','r18','config.yaml')  # 적절한 경로로 수정하세요
    config = parse_config(config_path)

    # 모델 로드
    model = load_model(config.model)
    fp_model = load_model(config.model)

    # 데이터 로드
    train_loader, val_loader = load_data(**config.data)
    cali_data = next(iter(train_loader))[0]  # 일부 데이터를 가져옵니다.

    # 모델 양자화
    print("Quantizing model...")
    model = quantize_model(model, config.quant)  # 양자화 적용
    # 테스트를 위해 reconstruction_with_mfm_gd_loss 함수를 호출
    print("Starting reconstruction process...")

    # 이 함수 안에서 module의 값이 어떻게 변경되는지 보기 위해 print문 추가
    for name, module in model.named_children():
        print(f"Before reconstruction: {name} -> {module}")

    for name, module in model.named_children():
        if isinstance(module, QuantizedBlock) or isinstance(module, QuantizedLayer):
            print(f"Starting reconstruction process for: {name} -> {module}")
            reconstruction_with_mfm_gd_loss(model, fp_model, module, getattr(fp_model, name), cali_data, config)


    # reconstruction 후의 module 값 확인
    for name, module in model.named_children():
        print(f"After reconstruction: {name} -> {module}")

if __name__ == "__main__":
    test_module_value()

import numpy as np
import torch
import torch.nn as nn
import logging
from imagenet_utils import DataSaverHook, StopForwardException
from quantize.quantized_module import QuantizedModule, QuantizedBlock, QuantizedLayer
from quantize.quant import LSQFakeQuantize, LSQPlusFakeQuantize, QuantizeBase
logger = logging.getLogger('qdrop')

'''
recon.py => reconsturction 수행 + adaround 수행 
1. 양자화된 입력 및 출력 저장 
2. fp32 모델과 quantized된 모델 간의 차이를 줄이기 위한 reconstruction
3. scale, zero point 와 같이 학습 가능한 파라미터에 대한 최적화 수행하여 양자화된 모델의 성능을 개선 
'''

# ptmq : Multi-Bit Feature Mixture 구현 

"""
    우선 각 비트 그룹 별로 tuning된 scaling factor를 적용한 후 feature를 추출하고, 양자화된 모델의 feature와 fp32 모델의 feature 를 섞어주기 
    bit_configs  [(2비트 설정, low), (4비트 설정, middle), (6비트 설정, high)]
"""
def get_bit_feature(data, bit):
  
    qmin = 0
    qmax = 2 ** bit - 1
    min_val = data.min().item()
    max_val = data.max().item()

    if max_val == min_val:
        scale = 1.0
    else:
        scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale
    zero_point = int(round(zero_point))
    quantized_data = ((data / scale).round() + zero_point).clamp(qmin, qmax)
    dequantized_data = (quantized_data - zero_point) * scale
    return dequantized_data
 

def multi_bit_mix(f_l,f_m, f_h, fp_features, p=0.5 ):

    # mix weight 설정 (일단은 균등하게...  )
    lambda_1, lambda_2, lambda_3 = 1/3, 1/3, 1/3

    mixed_quant_features = lambda_1 * f_l + lambda_2 * f_m + lambda_3 * f_h

    print(f"f_l shape: {f_l.shape}, f_m shape: {f_m.shape}, f_h shape: {f_h.shape}, fp_features shape: {fp_features.shape}")

    # fp_feature와 mixed_quant_feature를 p에 따라 섞기

    if torch.rand(1).item() < p :
        return mixed_quant_features
    else:
        return (1-p) * mixed_quant_features + p * fp_features


def gd_loss(f_h, f_m, f_l, f_fp, gamma1, gamma2, gamma3):

    """ gd-loss를 계산하여 고비트 특징이 중-저 비트 특징을 감독하고 add oper 수행
    => fp f 가 한번 더 supervise """


    # high - middle mse 계산, high-low mse 계산

    mse_hm = torch.nn.functional.mse_loss(f_h , f_m)
    mse_hl = torch.nn.functional.mse_loss (f_h, f_l)

    # add 

    combined_feature = f_h + f_m + f_l


    # fp 와의 mse 계산 

    mse_fp = torch.nn.functional.mse_loss(combined_feature, f_fp)

    total_loss = gamma1 * mse_fp + gamma2 * mse_hm + gamma3 * mse_hl

    return total_loss


def save_inp_oup_data(model, module, cali_data: list, store_inp=False, store_oup=False, bs: int = 32, keep_gpu: bool = True):
   
    # model : 평가할 모델 
    # module : 입력/출력을 추적할 레이어 
    # cali_data: calibration용 데이터 셋 
    # store_inp : input data의 저장 여부 
    # store_oup : output data의 저장 여부 
    # bs : 배치 크기 
    # keep_gpu : 데이터를 GPU에 유지할 지 여부 


    device = next(model.parameters()).device # 모델의 parameter로부터 device(gpu, cpu)를 설정

    # 데이터 저장용 훅 (DataSaverHook)을 설정하여 지정된 레이어의 입력과 출력을 저장한다. 

    data_saver = DataSaverHook(store_input=store_inp, store_output=store_oup, stop_forward=True)
    handle = module.register_forward_hook(data_saver)
    cached = [[], []]

    with torch.no_grad(): # 모델을 calibration 데이터에 대해서 평가한다. 각 배치마다 데이터를 저장한다. 
        for i in range(int(cali_data.size(0) / bs)):
            try:
                _ = model(cali_data[i * bs: (i + 1) * bs].to(device))
            except StopForwardException: # 필요한만큼의 데이터를 저장한 후 계산을 중단시킨다. 
                pass

            # 입/출력 데이터를 gpu - cpu 에 남겨둘지 아님 저장 안할지 .. 
            if store_inp:
                if keep_gpu:
                    cached[0].append(data_saver.input_store[0].detach())
                else:
                    cached[0].append(data_saver.input_store[0].detach().cpu())
            if store_oup:
                if keep_gpu:
                    cached[1].append(data_saver.output_store.detach())
                else:
                    cached[1].append(data_saver.output_store.detach().cpu())

    if store_inp:
        cached[0] = torch.cat([x for x in cached[0]]) # 데이터 캐시를 최종적으로 병합한다. 
    if store_oup:
        cached[1] = torch.cat([x for x in cached[1]])
    handle.remove()
    torch.cuda.empty_cache() # 메모리 사용량 줄이기 위해서 캐시 비우기 
    return cached


class LinearTempDecay:
    # LinearTempDecay => 특정 시간 t에 대해 b값을 선형적으로 감소시킨다. 
    # b값은 regularization loss에서 사용된다. (반올림 손실)
    def __init__(self, t_max=20000, warm_up=0.2, start_b=20, end_b=2):

        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t)) # t에 따른 선형적으로 감소된 값 반환 


# reconsturction 과정에서 loss를 계산하는 함수. 
# 일정 시간 동안 두 가지 손실을 균형있게 줄인다. 

class LossFunction:
    r'''loss function to calculate mse reconstruction loss and relaxation loss
    use some tempdecay to balance the two losses.
    '''

    def __init__(self,
                 module: QuantizedModule,
                 weight: float = 1.,
                 iters: int = 20000,
                 b_range: tuple = (20, 2),
                 warm_up: float = 0.0,
                 p: float = 2.):

        self.module = module
        self.weight = weight
        self.loss_start = iters * warm_up
        self.p = p

        self.temp_decay = LinearTempDecay(iters, warm_up=warm_up,
                                          start_b=b_range[0], end_b=b_range[1]) # b값을 선형적으로 감소시키기 
        self.count = 0

    def __call__(self, pred, tgt):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :return: total loss function
        """

        # loss 계산을 위한 모듈과 파라미터를 초기화한다. 
        # temp_decay는 b값을 선형적으로 감소시킨다. 

        self.count += 1
        rec_loss = lp_loss(pred, tgt, p=self.p) # pred(예측값)과 tgt(정답)을 비교하여 재구성 손실을 계산한다.
        b = self.temp_decay(self.count) # 현재 시점의 b 값을 temp_decay를 통해 얻는다. 



        if self.count < self.loss_start:
            round_loss = 0
        else:
            round_loss = 0
            for layer in self.module.modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    round_vals = layer.weight_fake_quant.rectified_sigmoid()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        # 반올림 손실을 계산한다. 이 때, recified_sigmoid를 사용하여 오차를 최소화한다. 

        total_loss = rec_loss + round_loss # reconsturction 손실과 round 손실을 합산하여 총 손실을 계산한다. 
        if self.count % 500 == 0: # 일정 주기마다 로그로 기록한다. 
            logger.info('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss
    

def lp_loss(pred, tgt, p=2.0):
    """
    loss function
    """
    return (pred - tgt).abs().pow(p).sum(1).mean()


""" mfm 과 gd_loss 를 고려한 reconstruction 함수 """


def reconstruction_with_mfm_gd_loss(model, fp_model, module, fp_module, cali_data, config):
    device = next(module.parameters()).device

    # 데이터를 먼저 저장
    quant_inp, _ = save_inp_oup_data(model, module, cali_data, store_inp=True, store_oup=False, bs=config.quant.recon.batch_size, keep_gpu=config.quant.recon.keep_gpu)
    fp_inp, fp_oup = save_inp_oup_data(fp_model, fp_module, cali_data, store_inp=True, store_oup=True, bs=config.quant.recon.batch_size, keep_gpu=config.quant.recon.keep_gpu)

    w_para, a_para = [], []

    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight_quantizer = layer.weight_fake_quant
            weight_quantizer.init(layer.weight.data, config.quant.recon.round_mode)
            w_para += [weight_quantizer.alpha]
        if isinstance(layer, QuantizeBase) and 'post_act_fake_quantize' in name:
            layer.drop_prob = config.quant.recon.drop_prob
            if isinstance(layer, LSQFakeQuantize):
                a_para += [layer.scale]
            if isinstance(layer, LSQPlusFakeQuantize):
                a_para += [layer.scale]
                a_para += [layer.zero_point]

    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=config.quant.recon.scale_lr)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=config.quant.recon.iters, eta_min=0.)
    else:
        a_opt, a_scheduler = None, None

    w_opt = torch.optim.Adam(w_para)

    sz = quant_inp.size(0)

    for i in range(config.quant.recon.iters):
        idx = torch.randint(0, sz, (config.quant.recon.batch_size,))

        if config.quant.recon.drop_prob < 1.0:
            cur_quant_inp = quant_inp[idx].to(device)
            cur_fp_inp = fp_inp[idx].to(device)
            cur_inp = torch.where(torch.rand_like(cur_quant_inp) < config.quant.recon.drop_prob, cur_quant_inp, cur_fp_inp)
        else:
            cur_inp = quant_inp[idx].to(device)

        cur_fp_oup = fp_oup[idx].to(device)

# =============================================================================================================

        # MFM & GD-LOSS utilize 

        # block => high, mid, low 각각 quantize를 해서 feature를 뽑는다. 

        # 구현 과제 ... LOW _ MID _ HIGH BIT FEATURE 를 어떻게 뽑아야 좋을까??? 각 함수 다 만드러야 댐 
    # Generate quantized features at different bit widths

        f_l = get_bit_feature(cur_fp_oup, bit=2)
        f_m = get_bit_feature(cur_fp_oup, bit=4)
        f_h = get_bit_feature(cur_fp_oup, bit=6)
        f_fp = cur_fp_oup  # quantized 되지 않은 block의 output이 32비트 feature map 



        # MFM을 사용해 혼합된 특징 생성
        mixed_feature = multi_bit_mix(f_l, f_m, f_h, f_fp)
        print(f"Mixed feature shape: {mixed_feature.shape}")

        # 위에서 뽑은 mixed_feature를 Quantized Block 의 새로운 입력으로 사용 

        new_f_l = get_bit_feature(mixed_feature, bit=2)

        new_f_m = get_bit_feature(mixed_feature, bit=4)
      
        new_f_h = get_bit_feature(mixed_feature, bit=6)
        
        # 일단은 그냥 fp 로 가보자... 
        # GD-Loss 계산 => 여기에서 위의 mfm 에서 구한 mixed_feature를 input으로써 gd-loss 에 넣어줘야 한다. 추가하기 ... 차후에!!! input으로 넣어주고 거기에서 high,m,low bit feature 뽑기 

        # 2. 새로운 f_l, f_m, f_h를 사용하여 GD-Loss 계산(reconsturction loss)

        rec_loss = gd_loss(new_f_h, new_f_m, new_f_l, f_fp,
                       gamma1=config.quant.recon.gamma1, gamma2=config.quant.recon.gamma2, gamma3=config.quant.recon.gamma3)
        
    # ==========================================================================================================

        loss_func = LossFunction(module=module, weight=config.quant.recon.weight, iters=config.quant.recon.iters, b_range=config.quant.recon.b_range,
                             warm_up=config.quant.recon.warm_up)
        
        # 여기서부터 rounding loss 

        round_loss = 0
        for name, layer in module.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                round_vals = layer.weight_fake_quant.rectified_sigmoid()
                
                # Compute b using temp_decay logic
                b = loss_func.temp_decay(i)  # i is the current iteration

                round_loss += config.quant.recon.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()

        loss = rec_loss + round_loss

        if a_opt:
            a_opt.zero_grad()
        w_opt.zero_grad()
        loss.backward()
        w_opt.step()

        if a_opt:
            a_opt.step()
            a_scheduler.step()

    torch.cuda.empty_cache()

    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight_quantizer = layer.weight_fake_quant
            layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
            weight_quantizer.adaround = False
        if isinstance(layer, QuantizeBase) and 'post_act_fake_quantize' in name:
            layer.drop_prob = 1.0
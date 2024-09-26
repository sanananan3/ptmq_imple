# config.yaml 에서... 


quant:
    a_qconfig:
        quantizer: LSQFakeQuantize 
        observer: MSEObserver
        bit: 8
        symmetric: False
        ch_axis: -1
    w_qconfig:
        quantizer: AdaRoundFakeQuantize
        observer: MSEObserver
        bit: 8
        symmetric: False
        ch_axis: 0
    calibrate: 1024
    recon:
        batch_size: 32
        scale_lr: 4.0e-5
        warm_up: 0.2
        weight: 0.01
        iters: 20000
        b_range: [20, 2]
        keep_gpu: True
        round_mode: learned_hard_sigmoid
        drop_prob: 0.5
        gamma1: 0.3
        gamma2: 0.3
        gamma3: 0.3

model:                    # architecture details
    type: resnet18        # model name
    kwargs:
        num_classes: 1000
    path: "C:/Users/user/Desktop/ptmq-pytorch/ptmq/model/resnet18_imagenet.pth" 
data:
    path: "C:/Users/user/Desktop/ptmq-pytorch/ptmq/ImageNet"
    batch_size: 64
    num_workers: 4
    pin_memory: True
    input_size: 224
    test_resize: 256
process:
    seed: 1005


# 여기에서 model & data 의 path를 상대 경로로 지정하니 자꾸 오류가 떠서 절대 경로로 지정을 해주긴 하였으나 
# 만약에 clone을 해서 해당 코드들을 사용해야 한다면 상대 경로로 수정하거나, 본인의 로컬 컴퓨터의 절대 경로로 수정하길 바람... 
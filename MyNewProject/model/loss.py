import torch.nn.functional as F  # PyTorch의 함수형 API를 임포트

def nll_loss(output, target):
    """
    음의 로그 우도 손실 함수를 정의합니다.
    
    :param output: 모델의 예측 값 (출력)
    :param target: 실제 값 (타겟)
    :return: 음의 로그 우도 손실 값
    """
    return F.nll_loss(output, target)  # PyTorch의 nll_loss 함수를 사용하여 손실 값을 계산하고 반환
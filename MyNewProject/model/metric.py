import torch  # PyTorch 라이브러리 임포트

def accuracy(output, target):
    """
    정확도를 계산하는 함수.
    
    :param output: 모델의 예측 값 (출력)
    :param target: 실제 값 (타겟)
    :return: 정확도
    """
    with torch.no_grad():  # 그래디언트 계산을 하지 않도록 설정
        pred = torch.argmax(output, dim=1)  # 각 샘플에 대해 가장 높은 값을 가진 클래스 선택
        assert pred.shape[0] == len(target)  # 예측 값의 개수와 타겟의 개수가 동일한지 확인
        correct = 0  # 맞춘 개수를 초기화
        correct += torch.sum(pred == target).item()  # 예측이 맞은 개수 합산
    return correct / len(target)  # 정확도를 계산하여 반환

def top_k_acc(output, target, k=3):
    """
    Top-K 정확도를 계산하는 함수.
    
    :param output: 모델의 예측 값 (출력)
    :param target: 실제 값 (타겟)
    :param k: K 값 (기본값은 3)
    :return: Top-K 정확도
    """
    with torch.no_grad():  # 그래디언트 계산을 하지 않도록 설정
        pred = torch.topk(output, k, dim=1)[1]  # 각 샘플에 대해 상위 K개의 클래스를 선택
        assert pred.shape[0] == len(target)  # 예측 값의 개수와 타겟의 개수가 동일한지 확인
        correct = 0  # 맞춘 개수를 초기화
        for i in range(k):  # 상위 K개의 클래스 중 하나라도 맞는지 확인
            correct += torch.sum(pred[:, i] == target).item()  # 예측이 맞은 개수 합산
    return correct / len(target)  # Top-K 정확도를 계산하여 반환
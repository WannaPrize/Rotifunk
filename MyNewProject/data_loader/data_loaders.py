from torchvision import datasets, transforms  # torchvision 라이브러리에서 데이터셋과 변환 모듈 임포트
from base import BaseDataLoader  # base 모듈에서 BaseDataLoader 클래스 임포트

class FashionMnistDataLoader(BaseDataLoader):
    """
    BaseDataLoader를 사용한 Fashion MNIST 데이터 로딩
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # 데이터 변환 설정: 텐서로 변환한 후 평균과 표준편차로 정규화
        trsfm = transforms.Compose([
            transforms.ToTensor(),  # 이미지를 텐서로 변환
            transforms.Normalize((0.2860,), (0.3530,))  # 평균 0.2860, 표준편차 0.3530으로 정규화
        ])
        self.data_dir = data_dir  # 데이터 디렉토리 설정
        # Fashion MNIST 데이터셋 로드: train 인자로 학습 데이터인지 테스트 데이터인지 결정
        self.dataset = datasets.FashionMNIST(
            self.data_dir, 
            train=training, 
            download=True,  # 인터넷에서 데이터셋 다운로드
            transform=trsfm  # 변환 적용
        )
        # 부모 클래스(BaseDataLoader)의 초기화 메서드 호출
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
import argparse  # 커맨드 라인 인자를 처리하기 위한 argparse 모듈 임포트
import collections  # 네임드 튜플을 사용하기 위한 collections 모듈 임포트
import torch  # PyTorch 라이브러리 임포트
import numpy as np  # NumPy 라이브러리 임포트
import data_loader.data_loaders as module_data  # 데이터 로더 모듈 임포트
import model.loss as module_loss  # 손실 함수 모듈 임포트
import model.metric as module_metric  # 메트릭 모듈 임포트
import model.model as module_arch  # 모델 아키텍처 모듈 임포트
from parse_config import ConfigParser  # 설정 파일을 파싱하기 위한 ConfigParser 임포트
from trainer import Trainer  # 학습을 위한 Trainer 클래스 임포트
from utils import prepare_device  # 장치 설정을 위한 함수 임포트

# 재현성을 위해 랜덤 시드 고정
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')  # 로거 설정

    # 데이터 로더 인스턴스 설정
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()  # 검증 데이터 로더 분리

    # 모델 아키텍처 설정 및 출력
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # GPU 학습을 위한 장치 설정
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)  # 다중 GPU 설정

    # 손실 함수 및 메트릭 함수 가져오기
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # 옵티마이저 및 학습률 스케줄러 설정 (스케줄러 비활성화를 위해 관련 줄을 삭제할 수 있음)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer) if 'lr_scheduler' in config.config else None

    # 트레이너 초기화 및 학습 시작
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()  # 학습 시작

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')  # argparse를 사용하여 커맨드 라인 인자 파서 생성
    args.add_argument('-c', '--config', default=None, type=str,
                      help='설정 파일 경로 (기본값: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='최신 체크포인트 경로 (기본값: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='사용할 GPU의 인덱스 (기본값: all)')

    # 기본 설정 파일의 값을 커스텀 CLI 옵션으로 변경하기 위해 CustomArgs 추가
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),  # 학습률 인자 추가
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')  # 배치 크기 인자 추가
    ]
    config = ConfigParser.from_args(args, options)  # 설정 파일 및 커스텀 옵션 파싱
    main(config)  # 메인 함수 실행
import os  # 운영체제 관련 작업을 위한 모듈 임포트
import logging  # 로깅 모듈 임포트
from pathlib import Path  # 파일 및 디렉토리 경로 작업을 위한 모듈 임포트
from functools import reduce, partial  # 함수형 프로그래밍 도구 임포트
from operator import getitem  # 시퀀스 내 아이템을 가져오는 함수 임포트
from datetime import datetime  # 날짜와 시간 작업을 위한 모듈 임포트
from logger import setup_logging  # 로깅 설정을 위한 함수 임포트
from utils import read_json, write_json  # JSON 파일 읽기/쓰기를 위한 함수 임포트

class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        설정 파일을 파싱하는 클래스. 학습을 위한 하이퍼파라미터, 모듈 초기화, 체크포인트 저장 및 로깅 모듈을 처리합니다.
        :param config: 설정 및 하이퍼파라미터를 포함하는 딕셔너리. 예를 들어 `config.json` 파일의 내용.
        :param resume: 체크포인트 파일 경로.
        :param modification: 설정 딕셔너리에서 특정 값을 대체할 위치와 값을 지정하는 딕셔너리.
        :param run_id: 학습 프로세스를 위한 고유 식별자. 기본값으로 타임스탬프 사용.
        """
        # 설정 파일을 로드하고 수정 사항 적용
        self._config = _update_config(config, modification)
        self.resume = resume

        # 학습된 모델과 로그를 저장할 디렉토리 설정
        save_dir = Path(self.config['trainer']['save_dir'])

        exper_name = self.config['name']
        if run_id is None:  # 기본값으로 타임스탬프 사용
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id

        # 체크포인트와 로그를 저장할 디렉토리 생성
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # 업데이트된 설정 파일을 체크포인트 디렉토리에 저장
        write_json(self.config, self.save_dir / 'config.json')

        # 로깅 모듈 설정
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=''):
        """
        CLI 인자에서 이 클래스를 초기화합니다. 학습 및 테스트에서 사용됩니다.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        if args.config and resume:
            # 새로운 설정 파일로 업데이트 (파인튜닝 시 사용)
            config.update(read_json(args.config))

        # 커스텀 CLI 옵션을 딕셔너리로 파싱
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)

    def init_obj(self, name, module, *args, **kwargs):
        """
        설정 파일에서 'type'으로 지정된 함수 핸들을 찾아서 해당 인자로 초기화된 인스턴스를 반환합니다.
        예:
        `object = config.init_obj('name', module, a, b=1)`
        는
        `object = module.name(a, b=1)`
        와 동등합니다.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), '설정 파일에 있는 kwargs를 덮어쓰는 것은 허용되지 않습니다'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        설정 파일에서 'type'으로 지정된 함수 핸들을 찾아서 주어진 인자로 고정된 함수를 반환합니다.
        예:
        `function = config.init_ftn('name', module, a, b=1)`
        는
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`
        와 동등합니다.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), '설정 파일에 있는 kwargs를 덮어쓰는 것은 허용되지 않습니다'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """일반적인 딕셔너리처럼 항목에 접근합니다."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity 옵션 {}는 유효하지 않습니다. 유효한 옵션은 {}입니다.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # 읽기 전용 속성 설정
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# 커스텀 CLI 옵션으로 설정 딕셔너리를 업데이트하는 도우미 함수
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if (flg.startswith('--')):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """키 시퀀스로 트리 내의 중첩된 객체에 값을 설정합니다."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """키 시퀀스로 트리 내의 중첩된 객체에 접근합니다."""
    return reduce(getitem, keys, tree)
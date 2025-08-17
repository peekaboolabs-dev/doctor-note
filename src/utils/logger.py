"""
로깅 설정 모듈
"""

import logging
import sys
from pathlib import Path


def setup_logger(name, log_file=None, level=logging.DEBUG):
    """
    로거 설정

    Args:
        name: 로거 이름
        log_file: 로그 파일 경로 (None이면 콘솔만 출력)
        level: 로그 레벨

    Returns:
        logger: 설정된 로거 객체
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 이미 핸들러가 있으면 제거
    if logger.hasHandlers():
        logger.handlers.clear()

    # 포맷터 생성
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (옵션)
    if log_file:
        # 로그 디렉토리 생성
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name):
    """
    기존 로거 가져오기

    Args:
        name: 로거 이름

    Returns:
        logger: 로거 객체
    """
    return logging.getLogger(name)

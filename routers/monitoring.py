
from fastapi import APIRouter, HTTPException
from typing import List
import os

router = APIRouter()

LOGS_DIR = "logs"

def read_log_file(file_path: str) -> List[str]:
    """로그 파일을 읽어 마지막 5줄을 리스트로 반환"""
    if not os.path.exists(file_path):
        # 로그 파일이 없으면 빈 리스트 반환
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # 마지막 5줄만 반환
        return lines[-5:]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading log file: {str(e)}")

@router.get("/api/monitoring/api", tags=["Monitoring"], response_model=List[str])
async def get_api_monitoring_logs():
    """API 모니터링 로그를 반환"""
    log_file = os.path.join(LOGS_DIR, "api_monitoring.log")
    return read_log_file(log_file)

@router.get("/api/monitoring/ml", tags=["Monitoring"], response_model=List[str])
async def get_ml_monitoring_logs():
    """ML 모니터링 로그를 반환"""
    log_file = os.path.join(LOGS_DIR, "ml_monitoring.log")
    return read_log_file(log_file)

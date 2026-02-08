
import time
import logging
import json
from fastapi import Request

# API 모니터링 로거 설정
api_logger = logging.getLogger('api_monitoring')
api_logger.setLevel(logging.INFO)
api_handler = logging.FileHandler("logs/api_monitoring.log")
api_handler.setFormatter(logging.Formatter('%(message)s'))
api_logger.addHandler(api_handler)

async def log_requests(request: Request, call_next):
    """
    API 요청 및 응답에 대한 정보를 로깅하는 미들웨어
    """
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000  # 밀리초 단위
    
    log_data = {
        "path": request.url.path,
        "method": request.method,
        "status_code": response.status_code,
        "process_time_ms": round(process_time, 2)
    }
    
    api_logger.info(json.dumps(log_data))
    
    return response

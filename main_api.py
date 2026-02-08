import asyncio
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime
import os
import httpx
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# LLM 모델과 토크나이저, 장치를 저장할 전역 변수
llm_model = None
llm_tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# 로깅 기본 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# .env 파일 로드
load_dotenv()
from fastapi.staticfiles import StaticFiles

from routers.patient_info import api as patient_info_api, ml
from routers import page3
from routers import monitoring as monitoring_router
from core import database
from core.monitoring import log_requests
from routers.patient_info.ml import start_training_scheduler

# AI 보고서 생성 요청 모델
class AIReportRequest(BaseModel):
    patientName: str
    patientId: str
    severity: int
    testTime: str
    hospitalName: str
    hospitalAddress: str
    hospitalPhone: str
    medicalData: dict

app = FastAPI(
    title="Sep-Time API",
    version="1.0.0"
)

# 미들웨어 추가
app.middleware("http")(log_requests)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    # 데이터베이스 연결
    database.connect()
    
    # 기존 스케줄러 시작 로직 (정리)
    try:
        logging.info("LSTM 모델 학습 스케줄러 시작 시도...")
        session = database.get_db_session()
        # ml.start_training_scheduler(session)
        logging.info("LSTM 모델 학습 스케줄러가 시작되었습니다.")
    except Exception as e:
        logging.error(f"LSTM 스케줄러 시작 실패: {e}")

    # 로컬 LLM 모델 로드
    # global llm_model, llm_tokenizer
    # model_name = "google/gemma-2b-it"
    # cache_dir = "hf_models"  # 모델을 저장할 로컬 디렉토리

    # logging.info(f"'{model_name}' 모델 로드를 시작합니다. (저장위치: {cache_dir})")
    # logging.info(f"사용 장치: {device}")

    # try:
    #     # 환경변수에서 토큰 읽기
    #     hf_token = os.getenv("hf_token")
    #     if not hf_token:
    #         logging.warning("Hugging Face 토큰(hf_token)이 환경변수에 설정되지 않았습니다. Gated 모델 접근 시 에러가 발생할 수 있습니다.")

    #     llm_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)
    #     llm_model = AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         cache_dir=cache_dir,
    #         device_map=device,
    #         token=hf_token
    #     )
    #     logging.info(f"'{model_name}' 모델 로드 완료.")
    # except Exception as e:
    #     logging.error(f"모델 로드 중 에러 발생: {e}")

    # LSTM 모델 학습 스케줄러 시작
    try:
        loop = asyncio.get_running_loop()
        session_factory = database.get_session_factory()
        scheduler_thread = ml.start_training_scheduler(session_factory, loop)
        print("LSTM 모델 학습 스케줄러가 시작되었습니다.")
    except Exception as e:
        print(f"스케줄러 시작 실패: {e}")

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

app.include_router(patient_info_api.router)
app.include_router(page3.router)
app.include_router(monitoring_router.router)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/db-health")
async def db_health(session: AsyncSession = Depends(database.get_db_session)):
    try:
        result = await session.execute(text("SELECT 1"))
        return {"database_status": "connected", "result": result.scalar()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schedule-status")
async def get_schedule_status():
    """스케줄러 상태 확인"""
    return {
        "status": "active",
        "schedule": "8시간마다 LSTM 모델 학습",
        "message": "스케줄러가 백그라운드에서 실행 중입니다."
    }

@app.post("/api/generate-transfer-report", tags=["AI"])
async def generate_transfer_report(request: AIReportRequest):
    """
    환자 정보를 바탕으로 AI 전원 의뢰서 생성 (로컬 모델 사용)
    """
    try:
        if not llm_model or not llm_tokenizer:
            logging.error("모델이 로드되지 않았습니다. 서버 시작 로그를 확인하세요.")
            raise HTTPException(status_code=500, detail="모델이 준비되지 않았습니다.")

        logging.info("=== AI 보고서 생성 요청 (로컬 모델) ===")
        logging.info(f"환자: {request.patientName} ({request.patientId})")

        # 중증도 텍스트
        def get_severity_text(severity):
            if severity >= 8:
                return '중증'
            elif severity >= 5:
                return '중등도'
            else:
                return '경증'

        # 프롬프트 구성
        prompt = f"""당신은 의료 전문가입니다. 다음 환자 정보를 바탕으로 환자 전원 의뢰서를 작성해주세요.

【환자 기본 정보】
- 환자명: {request.patientName}
- 환자번호: {request.patientId}
- 중증도: {request.severity}/10 ({get_severity_text(request.severity)})
- 검사 시점: {request.testTime}

【이송 의료기관】
- 기관명: {request.hospitalName}
- 소재지: {request.hospitalAddress}
- 연락처: {request.hospitalPhone}

【현재 상태 및 검사 소견】
- D-Dimer (D-이합체): {request.medicalData.get('d_dimer', 0)} μg/mL (정상 범위: <0.5)
- LDH (젖산 탈수소효소): {request.medicalData.get('ldh', 0)} U/L (정상 범위: 140-280)
- Creatinine (크레아티닌): {request.medicalData.get('creatinine', 0)} mg/dL (정상 범위: 0.7-1.3)
- Hemoglobin (혈색소): {request.medicalData.get('hemoglobin', 0)} g/dL (정상 범위: 13-17)
- Lymphocytes (림프구): {request.medicalData.get('lymphocytes', 0)}% (정상 범위: 20-40)
- Neutrophils (호중구): {request.medicalData.get('neutrophils', 0)}% (정상 범위: 40-75)
- Platelet Count (혈소판): {request.medicalData.get('platelet_count', 0)} /μL (정상 범위: 150,000-450,000)
- WBC Count (백혈구): {request.medicalData.get('wbc_count', 0)} /μL (정상 범위: 4,000-11,000)
- hs-CRP (고감도 C-반응 단백): {request.medicalData.get('hs_crp', 0)} mg/L (정상 범위: <3)

【요청사항】
위 정보를 바탕으로 다음 형식의 전문적인 환자 전원 의뢰서를 작성해주세요:

1. 환자 기본 정보 (이름, 환자번호, 중증도)
2. 이송 의료기관 정보
3. 현재 상태 및 검사 소견 요약
4. 전원 사유 및 임상적 판단 (검사 수치를 근거로 상세히 작성)
5. 특이사항 및 주의사항 (이송 중 주의사항 포함)

의료 전문 용어를 사용하되, 명확하고 간결하게 작성해주세요."""

        # Gemma 프롬프트 템플릿
        full_prompt = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

        logging.info("로컬 모델으로 텍스트 생성을 시작합니다.")

        # 토크나이저로 프롬프트를 인코딩
        inputs = llm_tokenizer(full_prompt, return_tensors="pt").to(device)

        # 모델로 텍스트 생성
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=1024,
        )
        
        # 생성된 텍스트 디코딩 (입력 프롬프트 제외)
        result_text = llm_tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        ai_content = result_text.strip()

        logging.info("=== AI 보고서 생성 완료 ===")
        logging.info(f"생성된 내용 길이: {len(ai_content)} 문자")

        return {
            "status": "success",
            "report_content": ai_content,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logging.exception("AI 보고서 생성 중 예상치 못한 에러 발생")
        raise HTTPException(status_code=500, detail=str(e))

# Static files mounting
app.mount("/", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
    
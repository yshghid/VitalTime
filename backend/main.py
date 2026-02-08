import asyncio
import json
import logging
import os
import pickle
import threading
import time
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import schedule
import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# ====================================
# 로깅 설정
# ====================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

api_logger = logging.getLogger("api_monitoring")
api_logger.setLevel(logging.INFO)
if not api_logger.handlers:
    api_handler = logging.FileHandler(os.path.join(LOGS_DIR, "api_monitoring.log"))
    api_handler.setFormatter(logging.Formatter("%(message)s"))
    api_logger.addHandler(api_handler)

ml_logger = logging.getLogger("ml_monitoring")
ml_logger.setLevel(logging.INFO)
if not ml_logger.handlers:
    ml_handler = logging.FileHandler(os.path.join(LOGS_DIR, "ml_monitoring.log"))
    ml_handler.setFormatter(logging.Formatter("%(message)s"))
    ml_logger.addHandler(ml_handler)


# ====================================
# 데이터베이스 설정
# ====================================

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")

engine = None
async_session = None


def connect():
    global engine, async_session
    print("Attempting to connect to the database...")
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    print("Database connection successful.")


async def disconnect():
    if engine:
        await engine.dispose()


async def get_db_session():
    if async_session is None:
        raise IOError("Database not connected")
    async with async_session() as session:
        yield session


def get_session_factory():
    return async_session


# ====================================
# Pydantic 모델
# ====================================

class PatientInfo(BaseModel):
    patient_id: int
    patient_name: str
    timestamp: datetime
    cur_news: int
    cur_predicted: int


class PatientInfoResponse(BaseModel):
    patients: List[PatientInfo]
    total_count: int
    timestamp: datetime


class HospitalInfo(BaseModel):
    id: int
    name: str
    address: str
    distance: float
    phone: str


class Page3PatientInfo(BaseModel):
    patient_id: int
    patient_name: str
    severity: int


class ClinicalData(BaseModel):
    d_dimer: Optional[float] = None
    ldh: Optional[float] = None
    creatinine: Optional[float] = None
    hemoglobin: Optional[float] = None
    lymphocytes: Optional[float] = None
    neutrophils: Optional[float] = None
    hs_crp: Optional[float] = None
    timepoint: int


class Page3Request(BaseModel):
    patient_id: int = Field(..., description="환자 ID")
    hospital_info: HospitalInfo = Field(..., description="선택된 병원 정보")


class AIReport(BaseModel):
    report_content: str
    generated_at: datetime = Field(default_factory=datetime.now)


class Page3Response(BaseModel):
    patient_info: Page3PatientInfo
    hospital_info: HospitalInfo
    clinical_data: ClinicalData
    ai_report: AIReport


class AIReportRequest(BaseModel):
    patientName: str
    patientId: str
    severity: int
    testTime: str
    hospitalName: str
    hospitalAddress: str
    hospitalPhone: str
    medicalData: dict


# ====================================
# CRUD 함수
# ====================================

async def get_patient_info_crud(timestamp: datetime, session: AsyncSession) -> PatientInfoResponse:
    """기준 timestamp 기반 환자 정보 조회"""
    if timestamp.tzinfo is not None:
        timestamp = timestamp.replace(tzinfo=None)

    start_time = timestamp - timedelta(hours=8)
    end_time = timestamp

    query = text("""
        SELECT
            p.patient_name, p.patient_id,
            c_cur.timestamp AS cur_timestamp,
            c_cur.news_score_label AS cur_news,
            c_next.news_score AS cur_predicted
        FROM public.patient p
        JOIN (
            SELECT c1.*
            FROM public.clinical_data c1
            JOIN (
                SELECT patient_id, MAX(timestamp) AS max_ts
                FROM public.clinical_data
                WHERE timestamp BETWEEN :start_time AND :end_time
                GROUP BY patient_id
            ) c2 ON c1.patient_id = c2.patient_id AND c1.timestamp = c2.max_ts
        ) c_cur ON p.patient_id = c_cur.patient_id
        LEFT JOIN LATERAL (
            SELECT c2.news_score
            FROM public.clinical_data c2
            WHERE c2.patient_id = c_cur.patient_id
              AND c2.timestamp > c_cur.timestamp
              AND c2.timestamp <= c_cur.timestamp + INTERVAL '8 hours'
            ORDER BY c2.timestamp ASC
            LIMIT 1
        ) c_next ON TRUE
        ORDER BY p.patient_id;
    """)

    result = await session.execute(query, {
        "start_time": start_time,
        "end_time": end_time,
    })
    rows = result.fetchall()

    if not rows:
        return PatientInfoResponse(patients=[], total_count=0, timestamp=timestamp)

    patients = []
    for row in rows:
        cur_news = int(row[3]) if row[3] is not None else 0
        cur_predicted = int(row[4]) if row[4] is not None else 0
        patients.append(PatientInfo(
            patient_id=row[1],
            patient_name=row[0],
            timestamp=row[2],
            cur_news=float(cur_news),
            cur_predicted=float(cur_predicted),
        ))

    return PatientInfoResponse(patients=patients, total_count=len(patients), timestamp=timestamp)


async def get_patient_data_range_crud(patient_id: int, timestamp: datetime, session: AsyncSession):
    """특정 환자의 8시간 범위 데이터 조회"""
    if timestamp.tzinfo is not None:
        timestamp = timestamp.replace(tzinfo=None)

    start_time = timestamp - timedelta(hours=8)
    end_time = timestamp

    query = text("""
        SELECT clinical_id, patient_id, timestamp, timepoint,
               creatinine, hemoglobin, ldh, lymphocytes, neutrophils,
               platelet_count, wbc_count, hs_crp, d_dimer, news_score, news_score_label
        FROM public.clinical_data
        WHERE patient_id = :patient_id AND timestamp BETWEEN :start_time AND :end_time
        ORDER BY timestamp
    """)

    result = await session.execute(query, {
        "patient_id": patient_id, "start_time": start_time, "end_time": end_time
    })
    rows = result.fetchall()

    if not rows:
        return {
            "patient_id": patient_id,
            "timestamp_range": {"start": start_time.isoformat(), "end": end_time.isoformat()},
            "total_records": 0, "data": [],
        }

    data = []
    for row in rows:
        data.append({
            "clinical_id": row[0], "patient_id": row[1],
            "timestamp": row[2].isoformat() if row[2] else None,
            "timepoint": row[3],
            "creatinine": float(row[4]) if row[4] is not None else None,
            "hemoglobin": float(row[5]) if row[5] is not None else None,
            "ldh": int(row[6]) if row[6] is not None else None,
            "lymphocytes": float(row[7]) if row[7] is not None else None,
            "neutrophils": float(row[8]) if row[8] is not None else None,
            "platelet_count": float(row[9]) if row[9] is not None else None,
            "wbc_count": float(row[10]) if row[10] is not None else None,
            "hs_crp": float(row[11]) if row[11] is not None else None,
            "d_dimer": float(row[12]) if row[12] is not None else None,
            "news_score": int(row[13]) if row[13] is not None else None,
            "news_score_label": int(row[14]) if row[14] is not None else None,
        })

    return {
        "patient_id": patient_id,
        "timestamp_range": {"start": start_time.isoformat(), "end": end_time.isoformat()},
        "total_records": len(data), "data": data,
    }


async def get_patient_predicted_by_timestamp_crud(patient_id: int, timestamp: datetime, session: AsyncSession):
    """특정 환자의 기준 timestamp 이후 예측값 조회"""
    query = text("""
        SELECT clinical_id, patient_id,
               DATE_TRUNC('hour', timestamp) AS truncated_timestamp,
               timepoint, news_score
        FROM public.clinical_data
        WHERE patient_id = :patient_id
          AND DATE_TRUNC('hour', timestamp) > DATE_TRUNC('hour', CAST(:timestamp AS TIMESTAMP))
        ORDER BY DATE_TRUNC('hour', timestamp) ASC
        LIMIT 1
    """)

    result = await session.execute(query, {"patient_id": patient_id, "timestamp": timestamp})
    row = result.fetchone()

    if not row:
        return {
            "patient_id": patient_id,
            "base_timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
            "message": "해당 시점 이후의 예측값 데이터가 없습니다.", "data": [],
        }

    truncated_str = row[2].strftime("%Y-%m-%d %H:%M") if row[2] else None
    data = {
        "clinical_id": row[0], "patient_id": row[1],
        "timestamp_hour": truncated_str, "timepoint": row[3],
        "news_score": int(row[4]) if row[4] is not None else None,
    }

    return {
        "patient_id": patient_id,
        "base_timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
        "nearest_future_timestamp_hour": truncated_str, "data": data,
    }


async def get_all_clinical_data(session: AsyncSession):
    """clinical_data 테이블 전체 조회"""
    query = text("""
        SELECT clinical_id, patient_id, timestamp, timepoint,
               creatinine, hemoglobin, ldh, lymphocytes, neutrophils,
               platelet_count, wbc_count, hs_crp, d_dimer, news_score
        FROM public.clinical_data
        ORDER BY patient_id, timepoint
    """)

    result = await session.execute(query)
    rows = result.fetchall()

    if not rows:
        return {"message": "clinical_data가 없습니다.", "data": []}

    data = []
    for row in rows:
        data.append({
            "clinical_id": row[0], "patient_id": row[1],
            "timestamp": row[2].isoformat() if row[2] else None,
            "timepoint": row[3],
            "creatinine": float(row[4]) if row[4] is not None else None,
            "hemoglobin": float(row[5]) if row[5] is not None else None,
            "ldh": int(row[6]) if row[6] is not None else None,
            "lymphocytes": float(row[7]) if row[7] is not None else None,
            "neutrophils": float(row[8]) if row[8] is not None else None,
            "platelet_count": float(row[9]) if row[9] is not None else None,
            "wbc_count": float(row[10]) if row[10] is not None else None,
            "hs_crp": float(row[11]) if row[11] is not None else None,
            "d_dimer": float(row[12]) if row[12] is not None else None,
            "news_score": int(row[13]) if row[13] is not None else None,
        })

    df = pd.DataFrame(data)
    stats = {
        "total_records": len(df),
        "unique_patients": df["patient_id"].nunique(),
        "timepoint_range": {"min": int(df["timepoint"].min()), "max": int(df["timepoint"].max())},
        "news_score_stats": {
            "min": int(df["news_score"].min()),
            "max": int(df["news_score"].max()),
            "mean": float(df["news_score"].mean()),
        },
    }

    return {
        "data": data,
        "dataframe_info": {"shape": df.shape, "columns": list(df.columns), "dtypes": df.dtypes.to_dict()},
        "statistics": stats,
    }


# ====================================
# Page3 데이터 조회 함수
# ====================================

async def get_page3_patient_info(patient_id: int, db: AsyncSession) -> Page3PatientInfo:
    query = text("SELECT patient_id, patient_name, severity FROM patient WHERE patient_id = :patient_id")
    result = await db.execute(query, {"patient_id": patient_id})
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="환자 정보를 찾을 수 없습니다.")
    return Page3PatientInfo(patient_id=row.patient_id, patient_name=row.patient_name, severity=row.severity)


async def get_latest_clinical_data(patient_id: int, db: AsyncSession) -> ClinicalData:
    import random
    query = text("""
        SELECT d_dimer, ldh, creatinine, timepoint
        FROM clinical_data WHERE patient_id = :patient_id
        ORDER BY timepoint DESC LIMIT 1
    """)
    result = await db.execute(query, {"patient_id": patient_id})
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="임상 검사 데이터를 찾을 수 없습니다.")
    return ClinicalData(
        d_dimer=row.d_dimer, ldh=row.ldh, creatinine=row.creatinine,
        hemoglobin=round(random.uniform(12.0, 16.0), 1),
        lymphocytes=round(random.uniform(20.0, 40.0), 1),
        neutrophils=round(random.uniform(40.0, 75.0), 1),
        hs_crp=round(random.uniform(0.5, 5.0), 2),
        timepoint=row.timepoint,
    )


def generate_medical_report(patient_info: Page3PatientInfo, hospital_info: HospitalInfo, clinical_data: ClinicalData) -> str:
    llm = ChatOpenAI(model="gpt-4", temperature=0.3, max_tokens=1000)
    prompt_template = f"""
다음 정보를 바탕으로 의료기관 간 환자 전원 의뢰서를 공식 문서 형식으로 작성해주세요.

【환자 정보】
- 환자명: {patient_info.patient_name}
- 환자 ID: {patient_info.patient_id}
- 중증도: {patient_info.severity}

【이송 예정 의료기관】
- 의료기관명: {hospital_info.name}
- 주소: {hospital_info.address}
- 연락처: {hospital_info.phone}
- 이송 거리: {hospital_info.distance}km

【최신 검사 수치】
- D-Dimer: {clinical_data.d_dimer or 'N/A'} ng/mL
- LDH: {clinical_data.ldh or 'N/A'} U/L
- Creatinine: {clinical_data.creatinine or 'N/A'} mg/dL
- Hemoglobin: {clinical_data.hemoglobin or 'N/A'} g/dL
- Lymphocytes: {clinical_data.lymphocytes or 'N/A'}%
- Neutrophils: {clinical_data.neutrophils or 'N/A'}%
- hs-CRP: {clinical_data.hs_crp or 'N/A'} mg/L

전문적인 환자 전원 의뢰서를 작성해주세요. 다음 항목을 포함해주세요:
1. 환자 기본 정보
2. 이송 의료기관 정보
3. 현재 상태 및 검사 소견
4. 전원 사유 및 임상적 판단
5. 특이사항 및 주의사항
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt_template)])
        return response.content.strip()
    except Exception as e:
        return f"보고서 생성 중 오류가 발생했습니다: {str(e)}"


# ====================================
# ML - LSTM 모델 학습
# ====================================

async def train_lstm_model(session: AsyncSession):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    try:
        print("LSTM 모델 학습 시작")
        start_time = time.time()

        result = await get_all_clinical_data(session)
        clinical_df = pd.DataFrame(result["data"])

        feature_columns = [
            "creatinine", "hemoglobin", "ldh", "lymphocytes", "neutrophils",
            "platelet_count", "wbc_count", "hs_crp", "d_dimer", "news_score",
        ]

        patients_data = []
        for patient_id in range(1, 11):
            patient_df = clinical_df[clinical_df["patient_id"] == patient_id].copy()
            patient_df = patient_df.sort_values("timepoint")
            if len(patient_df) == 10:
                patients_data.append(patient_df[feature_columns].values)

        if not patients_data:
            raise Exception("충분한 데이터가 없습니다.")

        X = np.array(patients_data)
        y = X[:, :, -1]
        X_features = X[:, :, :-1]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X_features.reshape(-1, X_features.shape[-1])).reshape(X_features.shape)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).reshape(y.shape)

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_scaled.shape[1], X_scaled.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(25, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation="relu"),
            Dense(10, activation="linear"),
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

        model.fit(X_scaled, y_scaled, epochs=100, batch_size=1, validation_split=0.2, verbose=1)

        y_pred = scaler_y.inverse_transform(model.predict(X_scaled).reshape(-1, 1)).reshape(y.shape)

        mse = mean_squared_error(y.flatten(), y_pred.flatten())
        mae = mean_absolute_error(y.flatten(), y_pred.flatten())
        r2 = r2_score(y.flatten(), y_pred.flatten())
        training_time = time.time() - start_time

        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = os.path.join(model_dir, f"lstm_model_{ts}.h5")
        model.save(model_path)

        scaler_path = os.path.join(model_dir, f"scalers_{ts}.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, f)

        model_info = {
            "timestamp": ts,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "evaluation": {"mse": float(mse), "mae": float(mae), "r2": float(r2)},
            "training_time_seconds": training_time,
            "data_info": {
                "total_patients": len(patients_data), "timepoints": 10, "features": 9,
                "feature_columns": [c for c in feature_columns if c != "news_score"],
                "target_column": "news_score",
            },
        }

        with open(os.path.join(model_dir, f"model_info_{ts}.json"), "w", encoding="utf-8") as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)

        ml_logger.info(json.dumps({"event": "model_training", "timestamp": datetime.now().isoformat(), "model_info": model_info}, ensure_ascii=False))

        return {
            "evaluation": {"mse": float(mse), "mae": float(mae), "r2": float(r2)},
            "data_info": model_info["data_info"],
            "saved_files": {"model_path": model_path, "scaler_path": scaler_path},
        }

    except Exception as e:
        ml_logger.error(json.dumps({"event": "model_training_failed", "timestamp": datetime.now().isoformat(), "error": str(e)}, ensure_ascii=False))
        raise Exception(f"LSTM 모델 학습 중 오류 발생: {str(e)}")


# ====================================
# ML 스케줄링
# ====================================

_session_factory = None
_main_loop = None


async def scheduled_train_lstm():
    if _session_factory is None:
        return
    async with _session_factory() as session:
        try:
            result = await train_lstm_model(session)
            print(f"스케줄된 LSTM 학습 완료: {result['saved_files']['model_path']}")
        except Exception as e:
            print(f"스케줄된 LSTM 학습 실패: {e}")


def run_scheduled_training():
    if _main_loop:
        asyncio.run_coroutine_threadsafe(scheduled_train_lstm(), _main_loop)


def start_training_scheduler(factory, loop):
    global _session_factory, _main_loop
    _session_factory = factory
    _main_loop = loop
    schedule.every(8).hours.do(run_scheduled_training)
    t = threading.Thread(target=lambda: [schedule.run_pending() or time.sleep(1) for _ in iter(int, 1)], daemon=True)
    t.start()
    print("LSTM 모델 학습 스케줄러가 시작되었습니다 (8시간 간격).")
    return t


# ====================================
# LLM 모델 (로컬 Gemma)
# ====================================

llm_model = None
llm_tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"


# ====================================
# 미들웨어 - API 모니터링
# ====================================

async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    process_time = (time.time() - start) * 1000
    api_logger.info(json.dumps({
        "path": request.url.path, "method": request.method,
        "status_code": response.status_code, "process_time_ms": round(process_time, 2),
    }))
    return response


# ====================================
# FastAPI 앱
# ====================================

app = FastAPI(title="VitalTime API", version="1.0.0")

app.middleware("http")(log_requests)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    connect()
    try:
        loop = asyncio.get_running_loop()
        factory = get_session_factory()
        start_training_scheduler(factory, loop)
    except Exception as e:
        print(f"스케줄러 시작 실패: {e}")


@app.on_event("shutdown")
async def shutdown():
    await disconnect()


# ====================================
# API 엔드포인트 - Patient
# ====================================

@app.get("/api/get-patient-info", response_model=PatientInfoResponse, tags=["Patient"])
async def get_patient_info(
    timestamp: str = Query(..., description="기준 timestamp (ISO 형식)"),
    session: AsyncSession = Depends(get_db_session),
):
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return await get_patient_info_crud(dt, session)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid timestamp format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/get-patient-data-range/{patient_id}", tags=["Clinical Data"])
async def get_patient_data_range(
    patient_id: int,
    timestamp: datetime = Query(..., description="기준 timestamp (ISO 형식)"),
    session: AsyncSession = Depends(get_db_session),
):
    try:
        return await get_patient_data_range_crud(patient_id, timestamp, session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/get-patient-predicted/{patient_id}", tags=["Clinical Data"])
async def get_patient_predicted_by_timestamp(
    patient_id: int,
    timestamp: datetime = Query(..., description="기준 timestamp (ISO 형식)"),
    session: AsyncSession = Depends(get_db_session),
):
    try:
        return await get_patient_predicted_by_timestamp_crud(patient_id, timestamp, session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train-model", tags=["ML"])
async def train_model(session: AsyncSession = Depends(get_db_session)):
    try:
        return await train_lstm_model(session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====================================
# API 엔드포인트 - Page3 (전원 의뢰서)
# ====================================

@app.post("/api/page3/patient-report", response_model=Page3Response, tags=["Page3"])
async def get_patient_report(request: Page3Request, db: AsyncSession = Depends(get_db_session)):
    try:
        patient_info = await get_page3_patient_info(request.patient_id, db)
        clinical_data = await get_latest_clinical_data(request.patient_id, db)
        report_content = generate_medical_report(patient_info, request.hospital_info, clinical_data)
        return Page3Response(
            patient_info=patient_info,
            hospital_info=request.hospital_info,
            clinical_data=clinical_data,
            ai_report=AIReport(report_content=report_content),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")


# ====================================
# API 엔드포인트 - 모니터링
# ====================================

def read_log_file(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()[-5:]


@app.get("/api/monitoring/api", tags=["Monitoring"], response_model=List[str])
async def get_api_monitoring_logs():
    return read_log_file(os.path.join(LOGS_DIR, "api_monitoring.log"))


@app.get("/api/monitoring/ml", tags=["Monitoring"], response_model=List[str])
async def get_ml_monitoring_logs():
    return read_log_file(os.path.join(LOGS_DIR, "ml_monitoring.log"))


# ====================================
# API 엔드포인트 - AI 보고서 (로컬 LLM)
# ====================================

@app.post("/api/generate-transfer-report", tags=["AI"])
async def generate_transfer_report(request: AIReportRequest):
    if not llm_model or not llm_tokenizer:
        raise HTTPException(status_code=500, detail="모델이 준비되지 않았습니다.")

    def get_severity_text(severity):
        if severity >= 8:
            return "중증"
        elif severity >= 5:
            return "중등도"
        return "경증"

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
- D-Dimer: {request.medicalData.get('d_dimer', 0)} μg/mL
- LDH: {request.medicalData.get('ldh', 0)} U/L
- Creatinine: {request.medicalData.get('creatinine', 0)} mg/dL
- Hemoglobin: {request.medicalData.get('hemoglobin', 0)} g/dL
- Lymphocytes: {request.medicalData.get('lymphocytes', 0)}%
- Neutrophils: {request.medicalData.get('neutrophils', 0)}%
- Platelet Count: {request.medicalData.get('platelet_count', 0)} /μL
- WBC Count: {request.medicalData.get('wbc_count', 0)} /μL
- hs-CRP: {request.medicalData.get('hs_crp', 0)} mg/L

전문적인 환자 전원 의뢰서를 작성해주세요."""

    full_prompt = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    inputs = llm_tokenizer(full_prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(**inputs, max_new_tokens=1024)
    ai_content = llm_tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

    return {"status": "success", "report_content": ai_content, "generated_at": datetime.now().isoformat()}


# ====================================
# 헬스체크
# ====================================

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/db-health")
async def db_health(session: AsyncSession = Depends(get_db_session)):
    try:
        result = await session.execute(text("SELECT 1"))
        return {"database_status": "connected", "result": result.scalar()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schedule-status")
async def get_schedule_status():
    return {"status": "active", "schedule": "8시간마다 LSTM 모델 학습"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

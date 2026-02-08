from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import dotenv
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# ====================================
# 환경 변수 로드
# ====================================
dotenv.load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

# ====================================
# FastAPI app
# ====================================
app = FastAPI(title="Patient Information Service", version="1.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================
# DB 연결
# ====================================
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession)

async def get_db_session():
    async with async_session() as session:
        yield session

# ====================================
# 데이터 모델
# ====================================
class PatientInfo(BaseModel):
    patient_name: str
    timestamp: datetime
    cur_news: int  # timepoint의 실제 news_score_label
    cur_predicted: int# timepoint+1의 예측 news_score

class PatientInfoResponse(BaseModel):
    patients: List[PatientInfo]
    total_count: int
    timestamp: datetime


# ====================================
# 유틸 함수
# ====================================


# ====================================
# API 엔드포인트
# ====================================

# --- 환자 정보 조회 ---
from fastapi import Query
from datetime import datetime, timedelta

@app.get("/api/get-patient-info", response_model=PatientInfoResponse, tags=["Patient"])
async def get_patient_info(
    timestamp: datetime = Query(..., description="기준 timestamp (ISO 형식 예: 2025-01-01T08:00:00)"),
    session: AsyncSession = Depends(get_db_session)
):
    """
    기준 timestamp를 전달받아 해당 timestamp - 8시간 ~ timestamp 사이에 측정값이 존재하는
    모든 환자의 cur_news(news_score_label)와 timestamp+8시간의 cur_predicted(news_score)를 반환
    """
    try:
        print(f"환자 정보 조회 시작: 기준 timestamp = {timestamp}")

        # 시간 범위 계산
        start_time = timestamp - timedelta(hours=8)
        end_time = timestamp
        future_time = timestamp + timedelta(hours=8)

        # SQL 쿼리
        # - 현재 구간(start_time~end_time)의 측정값 → cur_news (news_score_label)
        # - future_time의 예측값(news_score) → cur_predicted
        query = text("""
            SELECT 
                p.patient_name,
                p.patient_id,
                c_cur.timestamp AS cur_timestamp,
                COALESCE(c_cur.news_score_label, 0.0) AS cur_news,
                COALESCE(c_next.news_score, 0.0) AS cur_predicted
            FROM public.patient p
            JOIN public.clinical_data c_cur
                ON p.patient_id = c_cur.patient_id
                AND c_cur.timestamp BETWEEN :start_time AND :end_time
            LEFT JOIN public.clinical_data c_next
                ON p.patient_id = c_next.patient_id
                AND c_next.timestamp = :future_time
            ORDER BY p.patient_id
        """)

        result = await session.execute(query, {
            "start_time": start_time,
            "end_time": end_time,
            "future_time": future_time
        })
        rows = result.fetchall()

        if not rows:
            print("해당 구간에 해당하는 환자 데이터가 없습니다.")
            return PatientInfoResponse(patients=[], total_count=0, timestamp=timestamp)

        patients = []
        for row in rows:
            patient_name = row[0]
            patient_id = row[1]
            cur_timestamp = row[2]
            cur_news = row[3] if row[3] is not None else 0.0
            cur_predicted = row[4] if row[4] is not None else 0.0

            patients.append(PatientInfo(
                patient_name=patient_name,
                timestamp=cur_timestamp,
                cur_news=float(cur_news),
                cur_predicted=float(cur_predicted)
            ))

        print(f"환자 정보 조회 완료: {len(patients)}명")

        return PatientInfoResponse(
            patients=patients,
            total_count=len(patients),
            timestamp=timestamp
        )

    except Exception as e:
        print(f"환자 정보 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"환자 정보 조회 중 오류 발생: {str(e)}"
        )

# --- 특정 환자의 timepoint 9 데이터 조회 ---
from fastapi import Query
from datetime import datetime, timedelta


@app.get("/api/get-patient-data-range/{patient_id}", tags=["Clinical Data"])
async def get_patient_data_range(
        patient_id: int,
        timestamp: datetime = Query(..., description="기준 timestamp (ISO 형식 예: 2025-01-01T08:00:00)"),
        session: AsyncSession = Depends(get_db_session)
):
    """
    특정 환자의 특정 timestamp 기준 -8시간 ~ timestamp 사이의 데이터를 조회하여 Json으로 반환

    Args:
        patient_id (int): 조회할 환자 ID
        timestamp (datetime): 기준 시각 (ISO8601 형식)

    Returns:
        dict: 해당 환자의 임상 데이터 (Json 형태)
    """
    try:
        print(f"🔬 환자 {patient_id}번 데이터 조회 시작 — 기준 timestamp = {timestamp}")

        start_time = timestamp - timedelta(hours=8)
        end_time = timestamp

        # 특정 patient_id의 주어진 시간 범위에 해당하는 clinical_data 조회
        query = text("""
            SELECT 
                clinical_id,
                patient_id,
                timestamp,
                timepoint,
                creatinine,
                hemoglobin,
                ldh,
                lymphocytes,
                neutrophils,
                platelet_count,
                wbc_count,
                hs_crp,
                d_dimer,
                news_score,
                news_score_label
            FROM public.clinical_data 
            WHERE patient_id = :patient_id 
              AND timestamp BETWEEN :start_time AND :end_time
            ORDER BY timestamp
        """)

        result = await session.execute(query, {
            "patient_id": patient_id,
            "start_time": start_time,
            "end_time": end_time
        })
        rows = result.fetchall()

        if not rows:
            print(f"⚠환자 {patient_id}번의 해당 구간 데이터가 없습니다.")
            return {
                "patient_id": patient_id,
                "timestamp_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "total_records": 0,
                "data": []
            }

        # JSON으로 변환
        data = []
        for row in rows:
            data.append({
                'clinical_id': row[0],
                'patient_id': row[1],
                'timestamp': row[2].isoformat() if row[2] else None,
                'timepoint': row[3],
                'creatinine': float(row[4]) if row[4] is not None else None,
                'hemoglobin': float(row[5]) if row[5] is not None else None,
                'ldh': int(row[6]) if row[6] is not None else None,
                'lymphocytes': float(row[7]) if row[7] is not None else None,
                'neutrophils': float(row[8]) if row[8] is not None else None,
                'platelet_count': float(row[9]) if row[9] is not None else None,
                'wbc_count': float(row[10]) if row[10] is not None else None,
                'hs_crp': float(row[11]) if row[11] is not None else None,
                'd_dimer': float(row[12]) if row[12] is not None else None,
                'news_score': int(row[13]) if row[13] is not None else None,
                'news_score_label': int(row[14]) if row[14] is not None else None
            })

        print(f"환자 {patient_id}번 데이터 조회 완료: {len(data)}개 레코드")

        return {
            "patient_id": patient_id,
            "timestamp_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "total_records": len(data),
            "data": data
        }

    except Exception as e:
        print(f"환자 {patient_id}번 데이터 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"환자 {patient_id}번 데이터 조회 중 오류 발생: {str(e)}"
        )

from fastapi import Query
from datetime import datetime, timedelta

@app.get("/api/get-patient-predicted/{patient_id}", tags=["Clinical Data"])
async def get_patient_predicted_by_timestamp(
    patient_id: int,
    timestamp: datetime = Query(..., description="기준 timestamp (ISO 형식 예: 2025-01-01T08:00:00)"),
    session: AsyncSession = Depends(get_db_session)
):
    """
    특정 환자의 timestamp + 8시간 (예측값) 데이터 조회하여 Json으로 반환

    Args:
        patient_id (int): 환자 ID
        timestamp (datetime): 기준 시각 (ISO 형식)

    Returns:
        dict: 해당 환자의 예측값 clinical_data (Json 형태)
    """
    try:
        print(f"환자 {patient_id}번 예측값 조회 시작 — 기준 timestamp = {timestamp}")

        target_time = timestamp + timedelta(hours=8)

        # 특정 patient_id의 target_time에 해당하는 clinical_data 조회
        query = text("""
            SELECT 
                clinical_id,
                patient_id,
                timestamp,
                timepoint,
                creatinine,
                hemoglobin,
                ldh,
                lymphocytes,
                neutrophils,
                platelet_count,
                wbc_count,
                hs_crp,
                d_dimer,
                news_score
            FROM public.clinical_data 
            WHERE patient_id = :patient_id 
              AND timestamp = :target_time
        """)

        result = await session.execute(query, {
            "patient_id": patient_id,
            "target_time": target_time
        })
        rows = result.fetchall()

        if not rows:
            print(f"환자 {patient_id}번의 {target_time} 예측값 데이터가 없습니다.")
            return {
                "patient_id": patient_id,
                "target_timestamp": target_time.isoformat(),
                "message": f"해당 시점의 예측값 데이터가 없습니다.",
                "data": []
            }

        # JSON으로 변환
        data = []
        for row in rows:
            data.append({
                'clinical_id': row[0],
                'patient_id': row[1],
                'timestamp': row[2].isoformat() if row[2] else None,
                'timepoint': row[3],
                'creatinine': float(row[4]) if row[4] is not None else None,
                'hemoglobin': float(row[5]) if row[5] is not None else None,
                'ldh': int(row[6]) if row[6] is not None else None,
                'lymphocytes': float(row[7]) if row[7] is not None else None,
                'neutrophils': float(row[8]) if row[8] is not None else None,
                'platelet_count': float(row[9]) if row[9] is not None else None,
                'wbc_count': float(row[10]) if row[10] is not None else None,
                'hs_crp': float(row[11]) if row[11] is not None else None,
                'd_dimer': float(row[12]) if row[12] is not None else None,
                'news_score': int(row[13]) if row[13] is not None else None
            })

        print(f"환자 {patient_id}번 {target_time} 예측값 조회 완료: {len(data)}개 레코드")

        return {
            "patient_id": patient_id,
            "target_timestamp": target_time.isoformat(),
            "total_records": len(data),
            "data": data
        }

    except Exception as e:
        print(f"환자 {patient_id}번 예측값 데이터 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"환자 {patient_id}번 예측값 데이터 조회 중 오류 발생: {str(e)}"
        )


# --- timepoint 11 데이터 추가 ---
# @app.post("/api/postPatientData", tags=["Patient Data"])
# async def post_patient_data(patient_id: int, session: AsyncSession = Depends(get_db_session)):
#     """
#     특정 환자의 새로운 임상 데이터를 clinical_data 테이블에 추가
#     
#     Args:
#         patient_id: 환자 ID (cur_pid)
#         
#     Returns:
#         dict: 추가 결과 정보
#     """
#     try:
#         print(f"📝 환자 {patient_id}번 데이터 추가 시작")
#         
#         # t11_data 정의
#         t11_data = {
#             "Timepoint": "t11",
#             "Creatinine": 1.44,
#             "Hemoglobin": 13.2,
#             "LDH": 515.0,
#             "Lymphocytes": 2.58,
#             "Neutrophils": 3.52,
#             "Platelet count": 295.0,
#             "WBC count": 6.77,
#             "hs-CRP": 19.3,
#             "D-Dimer": 3.03,
#             "NEWS score": 10.0
#         }
#         
#         # clinical_data 테이블에 데이터 삽입
#         query = text("""
#             INSERT INTO public.clinical_data (
#                 patient_id, timestamp, timepoint,
#                 creatinine, hemoglobin, ldh, lymphocytes, neutrophils,
#                 platelet_count, wbc_count, hs_crp, d_dimer, news_score
#             ) VALUES (
#                 :patient_id, NOW(), 11,
#                 :creatinine, :hemoglobin, :ldh, :lymphocytes, :neutrophils,
#                 :platelet_count, :wbc_count, :hs_crp, :d_dimer, :news_score
#             )
#             RETURNING clinical_id
#         """)
#         
#         result = await session.execute(query, {
#             "patient_id": patient_id,
#             "creatinine": t11_data["Creatinine"],
#             "hemoglobin": t11_data["Hemoglobin"],
#             "ldh": int(t11_data["LDH"]),
#             "lymphocytes": t11_data["Lymphocytes"],
#             "neutrophils": t11_data["Neutrophils"],
#             "platelet_count": t11_data["Platelet count"],
#             "wbc_count": t11_data["WBC count"],
#             "hs_crp": t11_data["hs-CRP"],
#             "d_dimer": t11_data["D-Dimer"],
#             "news_score": int(t11_data["NEWS score"])
#         })
#         
#         clinical_id = result.fetchone()[0]
#         await session.commit()
#         
#         print(f"환자 {patient_id}번 데이터 추가 완료: clinical_id {clinical_id}")
#         
#         return {
#             "success": True,
#             "patient_id": patient_id,
#             "clinical_id": clinical_id,
#             "timepoint": 11,
#             "message": f"환자 {patient_id}번의 timepoint 11 데이터가 성공적으로 추가되었습니다.",
#             "added_data": t11_data
#         }
#         
#     except Exception as e:
#         print(f"환자 {patient_id}번 데이터 추가 실패: {e}")
#         await session.rollback()
#         raise HTTPException(
#             status_code=500,
#             detail=f"환자 {patient_id}번 데이터 추가 중 오류 발생: {str(e)}"
#         )

# ====================================
# 유틸 함수
# ====================================

async def get_all_clinical_data(session: AsyncSession):
    """
    clinical_data 테이블 전체 조회하여 DataFrame으로 반환
    
    Args:
        session: 데이터베이스 세션
        
    Returns:
        dict: 모든 임상 데이터 (DataFrame 형태)
    """
    try:
        print("전체 clinical_data 조회 시작")
        
        # clinical_data 테이블 전체 조회
        query = text("""
            SELECT 
                clinical_id,
                patient_id,
                timestamp,
                timepoint,
                creatinine,
                hemoglobin,
                ldh,
                lymphocytes,
                neutrophils,
                platelet_count,
                wbc_count,
                hs_crp,
                d_dimer,
                news_score
            FROM public.clinical_data 
            ORDER BY patient_id, timepoint
        """)
        
        result = await session.execute(query)
        rows = result.fetchall()
        
        if not rows:
            print("clinical_data가 없습니다.")
            return {"message": "clinical_data가 없습니다.", "data": []}
        
        # DataFrame으로 변환
        data = []
        for row in rows:
            data.append({
                'clinical_id': row[0],
                'patient_id': row[1],
                'timestamp': row[2].isoformat() if row[2] else None,
                'timepoint': row[3],
                'creatinine': float(row[4]) if row[4] is not None else None,
                'hemoglobin': float(row[5]) if row[5] is not None else None,
                'ldh': int(row[6]) if row[6] is not None else None,
                'lymphocytes': float(row[7]) if row[7] is not None else None,
                'neutrophils': float(row[8]) if row[8] is not None else None,
                'platelet_count': float(row[9]) if row[9] is not None else None,
                'wbc_count': float(row[10]) if row[10] is not None else None,
                'hs_crp': float(row[11]) if row[11] is not None else None,
                'd_dimer': float(row[12]) if row[12] is not None else None,
                'news_score': int(row[13]) if row[13] is not None else None
            })
        
        df = pd.DataFrame(data)
        
        print(f"전체 clinical_data 조회 완료: {len(df)}개 레코드")
        
        # 통계 정보 추가
        stats = {
            "total_records": len(df),
            "unique_patients": df['patient_id'].nunique(),
            "timepoint_range": {
                "min": int(df['timepoint'].min()),
                "max": int(df['timepoint'].max())
            },
            "news_score_stats": {
                "min": int(df['news_score'].min()),
                "max": int(df['news_score'].max()),
                "mean": float(df['news_score'].mean())
            }
        }
        
        return {
            "data": data,
            "dataframe_info": {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict()
            },
            "statistics": stats
        }
        
    except Exception as e:
        print(f"전체 clinical_data 조회 실패: {e}")
        raise Exception(f"전체 clinical_data 조회 중 오류 발생: {str(e)}")

async def train_lstm_model(session: AsyncSession):
    """
    clinical_data를 사용하여 LSTM 모델을 학습시키는 함수
    
    Args:
        session: 데이터베이스 세션
        
    Returns:
        dict: 학습 결과와 모델 정보
    """
    try:
        print("🧠 LSTM 모델 학습 시작")
        
        # 1. clinical_data 조회
        result = await get_all_clinical_data(session)
        clinical_df = pd.DataFrame(result['data'])
        
        print(f"📊 데이터 로드 완료: {len(clinical_df)}개 레코드")
        
        # 2. 피처 정의 (creatinine부터 news_score까지 10개)
        feature_columns = [
            'creatinine', 'hemoglobin', 'ldh', 'lymphocytes', 'neutrophils',
            'platelet_count', 'wbc_count', 'hs_crp', 'd_dimer', 'news_score'
        ]
        
        # 3. 데이터 전처리
        # patient_id별로 그룹화하여 시계열 데이터 구성
        patients_data = []
        for patient_id in range(1, 11):  # patient_id 1~10
            patient_df = clinical_df[clinical_df['patient_id'] == patient_id].copy()
            patient_df = patient_df.sort_values('timepoint')
            
            if len(patient_df) == 10:  # 10개 timepoint가 모두 있는 경우만
                # 피처 데이터 (10개 컬럼)
                features = patient_df[feature_columns].values
                patients_data.append(features)
        
        if len(patients_data) == 0:
            raise Exception("충분한 데이터가 없습니다. 각 환자마다 10개의 timepoint가 필요합니다.")
        
        # 4. 데이터 배열 변환
        X = np.array(patients_data)  # Shape: (10, 10, 10) - (환자수, timepoint, features)
        y = X[:, :, -1]  # news_score만 추출 - Shape: (10, 10)
        
        # 5. 피처와 타겟 분리
        X_features = X[:, :, :-1]  # news_score 제외한 9개 피처 - Shape: (10, 10, 9)
        y_target = y  # news_score - Shape: (10, 10)
        
        print(f"📈 데이터 형태: X_features {X_features.shape}, y_target {y_target.shape}")
        
        # 6. 데이터 정규화
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        # 3D -> 2D로 변환하여 정규화
        X_reshaped = X_features.reshape(-1, X_features.shape[-1])
        y_reshaped = y_target.reshape(-1, 1)
        
        X_scaled = scaler_X.fit_transform(X_reshaped)
        y_scaled = scaler_y.fit_transform(y_reshaped)
        
        # 다시 3D로 변환
        X_scaled = X_scaled.reshape(X_features.shape)
        y_scaled = y_scaled.reshape(y_target.shape)
        
        # 7. LSTM 모델 구성
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_scaled.shape[1], X_scaled.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(25, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(10, activation='linear')  # 10개 timepoint 예측
        ])
        
        # 8. 모델 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("🏗️ LSTM 모델 구성 완료")
        print(f"모델 구조:\n{model.summary()}")
        
        # 9. 모델 학습
        history = model.fit(
            X_scaled, y_scaled,
            epochs=100,
            batch_size=1,
            validation_split=0.2,
            verbose=1
        )
        
        # 10. 예측 및 평가
        y_pred_scaled = model.predict(X_scaled)
        
        # 역정규화
        y_pred_reshaped = y_pred_scaled.reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred_reshaped)
        y_pred = y_pred.reshape(y_target.shape)
        
        # 평가 지표 계산
        mse = mean_squared_error(y_target.flatten(), y_pred.flatten())
        mae = mean_absolute_error(y_target.flatten(), y_pred.flatten())
        r2 = r2_score(y_target.flatten(), y_pred.flatten())
        
        print(f"✅ 모델 학습 완료")
        print(f"📊 성능 지표:")
        print(f"   - MSE: {mse:.4f}")
        print(f"   - MAE: {mae:.4f}")
        print(f"   - R²: {r2:.4f}")
        
        # 11. 모델 로컬 저장
        import os
        from datetime import datetime
        
        # 모델 저장 디렉토리 생성
        model_dir = "saved_models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"📁 모델 저장 디렉토리 생성: {model_dir}")
        
        # 타임스탬프로 고유한 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 모델 저장 (pickle 형식)
        model_path = os.path.join(model_dir, f"lstm_model_{timestamp}.pkl")
        import pickle
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"💾 모델 저장 완료: {model_path}")
        
        model_info = {
            "timestamp": timestamp,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "evaluation": {
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2)
            },
            "data_info": {
                "total_patients": len(patients_data),
                "timepoints": 10,
                "features": 9,
                "feature_columns": feature_columns[:-1],
                "target_column": "news_score"
            },
            "model_architecture": {
                "input_shape": X_scaled.shape[1:],
                "layers": len(model.layers),
                "total_params": model.count_params()
            }
        }
        
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        print(f"💾 모델 정보 저장 완료: {model_info_path}")
        
        # 12. 결과 반환
        return {
            "model": model,
            "history": history.history,
            "evaluation": {
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2)
            },
            "data_info": {
                "total_patients": len(patients_data),
                "timepoints": 10,
                "features": 9,
                "feature_columns": feature_columns[:-1],  # news_score 제외
                "target_column": "news_score"
            },
            "scalers": {
                "feature_scaler": scaler_X,
                "target_scaler": scaler_y
            },
            "predictions": {
                "actual": y_target.tolist(),
                "predicted": y_pred.tolist()
            },
            "saved_files": {
                "model_path": model_path
            }
        }
        
    except Exception as e:
        print(f"❌ LSTM 모델 학습 실패: {e}")
        raise Exception(f"LSTM 모델 학습 중 오류 발생: {str(e)}")

# --- DB health ---
@app.get("/api/db-health")
async def db_health(session: AsyncSession = Depends(get_db_session)):
    try:
        result = await session.execute(text("SELECT 1"))
        return {"database_status": "connected", "result": result.scalar()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 서비스 health ---
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# ====================================
# 실행
# ====================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
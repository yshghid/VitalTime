import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from .crud import get_all_clinical_data
import os
import json
from datetime import datetime
import pickle
import asyncio
import schedule
import threading
import time
import logging

# ML 모니터링 로거 설정
ml_logger = logging.getLogger('ml_monitoring')
ml_logger.setLevel(logging.INFO)
# 핸들러가 이미 추가되었는지 확인하여 중복 추가 방지
if not ml_logger.handlers:
    ml_handler = logging.FileHandler('logs/ml_monitoring.log')
    ml_handler.setFormatter(logging.Formatter('%(message)s'))
    ml_logger.addHandler(ml_handler)

async def train_lstm_model(session: AsyncSession):
    """
    clinical_data를 사용하여 LSTM 모델을 학습시키는 함수
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    try:
        print("LSTM 모델 학습 시작")
        start_time = time.time()

        result = await get_all_clinical_data(session)
        clinical_df = pd.DataFrame(result['data'])

        print(f"데이터 로드 완료: {len(clinical_df)}개 레코드")

        feature_columns = [
            'creatinine', 'hemoglobin', 'ldh', 'lymphocytes', 'neutrophils',
            'platelet_count', 'wbc_count', 'hs_crp', 'd_dimer', 'news_score'
        ]

        patients_data = []
        for patient_id in range(1, 11):
            patient_df = clinical_df[clinical_df['patient_id'] == patient_id].copy()
            patient_df = patient_df.sort_values('timepoint')

            if len(patient_df) == 10:
                features = patient_df[feature_columns].values
                patients_data.append(features)

        if len(patients_data) == 0:
            raise Exception("충분한 데이터가 없습니다. 각 환자마다 10개의 timepoint가 필요합니다.")

        X = np.array(patients_data)
        y = X[:, :, -1]

        X_features = X[:, :, :-1]
        y_target = y

        print(f"데이터 형태: X_features {X_features.shape}, y_target {y_target.shape}")

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_reshaped = X_features.reshape(-1, X_features.shape[-1])
        y_reshaped = y_target.reshape(-1, 1)

        X_scaled = scaler_X.fit_transform(X_reshaped)
        y_scaled = scaler_y.fit_transform(y_reshaped)

        X_scaled = X_scaled.reshape(X_features.shape)
        y_scaled = y_scaled.reshape(y_target.shape)

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_scaled.shape[1], X_scaled.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(25, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(10, activation='linear')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        print("LSTM 모델 구성 완료")
        model.summary()

        history = model.fit(
            X_scaled, y_scaled,
            epochs=100,
            batch_size=1,
            validation_split=0.2,
            verbose=1
        )

        y_pred_scaled = model.predict(X_scaled)

        y_pred_reshaped = y_pred_scaled.reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred_reshaped)
        y_pred = y_pred.reshape(y_target.shape)

        mse = mean_squared_error(y_target.flatten(), y_pred.flatten())
        mae = mean_absolute_error(y_target.flatten(), y_pred.flatten())
        r2 = r2_score(y_target.flatten(), y_pred.flatten())
        
        end_time = time.time()
        training_time = end_time - start_time

        print(f"모델 학습 완료")
        print(f"성능 지표:")
        print(f"   - MSE: {mse:.4f}")
        print(f"   - MAE: {mae:.4f}")
        print(f"   - R²: {r2:.4f}")

        model_dir = "saved_models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"모델 저장 디렉토리 생성: {model_dir}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = os.path.join(model_dir, f"lstm_model_{timestamp}.h5")
        model.save(model_path)
        print(f"모델 저장 완료: {model_path}")
        
        scaler_path = os.path.join(model_dir, f"scalers_{timestamp}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)

        model_info_path = os.path.join(model_dir, f"model_info_{timestamp}.json")
        model_info = {
            "timestamp": timestamp,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "evaluation": {
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2)
            },
            "training_time_seconds": training_time,
            "data_info": {
                "total_patients": len(patients_data),
                "timepoints": 10,
                "features": 9,
                "feature_columns": [col for col in feature_columns if col != 'news_score'],
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
        print(f"모델 정보 저장 완료: {model_info_path}")

        # ML 모니터링 로그 기록
        log_entry = {
            "event": "model_training",
            "timestamp": datetime.now().isoformat(),
            "model_info": model_info
        }
        ml_logger.info(json.dumps(log_entry, ensure_ascii=False))

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
                "feature_columns": [col for col in feature_columns if col != 'news_score'],
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
                "model_path": model_path,
                "scaler_path": scaler_path,
                "model_info_path": model_info_path
            }
        }

    except Exception as e:
        print(f"LSTM 모델 학습 실패: {e}")
        # 실패 로그 기록
        log_entry = {
            "event": "model_training_failed",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }
        ml_logger.error(json.dumps(log_entry, ensure_ascii=False))
        raise Exception(f"LSTM 모델 학습 중 오류 발생: {str(e)}")



# ====================================
# 스케줄링 관련 함수들
# ====================================

# 전역 변수로 세션 팩토리 및 메인 이벤트 루프 관리
_session_factory = None
_main_loop = None

def set_session_factory(factory):
    """스케줄링용 세션 팩토리 설정"""
    global _session_factory
    _session_factory = factory

def set_main_loop(loop):
    """메인 이벤트 루프 설정"""
    global _main_loop
    _main_loop = loop

async def scheduled_train_lstm():
    """스케줄링된 LSTM 모델 학습 함수"""
    if _session_factory is None:
        print("스케줄링용 세션 팩토리가 설정되지 않았습니다.")
        return
    
    print("스케줄된 LSTM 모델 학습 시작")
    async with _session_factory() as session:
        try:
            result = await train_lstm_model(session)
            print(f"스케줄된 LSTM 모델 학습 완료: {result['saved_files']['model_path']}")
            current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"스케줄링 완료 : {current_timestamp}")
        except Exception as e:
            print(f"스케줄된 LSTM 모델 학습 실패: {e}")
            current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"스케줄링 완료 (실패) : {current_timestamp}")

def run_scheduled_training():
    """스케줄링된 학습을 실행하는 동기 함수"""
    if _main_loop is None:
        print("메인 이벤트 루프가 설정되지 않았습니다.")
        return
    
    # 메인 스레드의 이벤트 루프에서 코루틴을 스레드 안전하게 실행
    asyncio.run_coroutine_threadsafe(scheduled_train_lstm(), _main_loop)

def setup_training_schedule():
    """30초마다 LSTM 모델 학습을 실행하는 스케줄 설정"""
    print("LSTM 모델 학습 스케줄 설정 시작")
    schedule.every(8).hours.do(run_scheduled_training)
    print("LSTM 모델 학습 스케줄 설정 완료")
    print("   - 30초마다 실행")

def run_scheduler():
    """스케줄러를 백그라운드에서 실행"""
    print("스케줄러 백그라운드 실행 시작")
    while True:
        schedule.run_pending()
        time.sleep(1)

def start_training_scheduler(factory, loop):
    """LSTM 모델 학습 스케줄러 시작"""
    set_session_factory(factory)
    set_main_loop(loop)
    setup_training_schedule()
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    print("LSTM 모델 학습 스케줄러가 백그라운드에서 시작되었습니다.")
    return scheduler_thread

from datetime import datetime, timedelta
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
from .models import PatientInfo, PatientInfoResponse

async def get_patient_info(timestamp: datetime, session: AsyncSession) -> PatientInfoResponse:
    """
    기준 timestamp를 전달받아 해당 timestamp - 8시간 ~ timestamp 사이에 측정값이 존재하는
    모든 환자의 cur_news(news_score_label)와 timestamp+8시간의 cur_predicted(news_score)를 반환
    """
    print(f"환자 정보 조회 시작: 기준 timestamp = {timestamp}")

    # timezone 제거 (DB timestamp 비교용)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.replace(tzinfo=None)

    start_time = timestamp - timedelta(hours=8)
    end_time = timestamp
    future_time = timestamp + timedelta(hours=8)

    print(f"시간 범위: {start_time} ~ {end_time}, 미래: {future_time}")

    query = text("""
        SELECT
            p.patient_name,
            p.patient_id,
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
            ) c2
            ON c1.patient_id = c2.patient_id AND c1.timestamp = c2.max_ts
        ) c_cur
        ON p.patient_id = c_cur.patient_id
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

    try:
        result = await session.execute(query, {
            "start_time": start_time,
            "end_time": end_time,
            "future_time": future_time
        })
        rows = result.fetchall()
        print(f"쿼리 실행 완료: {len(rows)}개 행 반환")
    except Exception as e:
        print(f"쿼리 실행 중 에러: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

    if not rows:
        print("해당 구간에 해당하는 환자 데이터가 없습니다.")
        return PatientInfoResponse(patients=[], total_count=0, timestamp=timestamp)

    patients = []
    for row in rows:
        try:
            patient_name = row[0]
            patient_id = row[1]
            cur_timestamp = row[2]
            cur_news = int(row[3]) if row[3] is not None else 0 # 수정 필요
            cur_predicted = int(row[4]) if row[4] is not None else 0 # 수정 필요

            patients.append(PatientInfo(
                patient_id=patient_id,
                patient_name=patient_name,
                timestamp=cur_timestamp,
                cur_news=float(cur_news),
                cur_predicted=float(cur_predicted)
            ))
        except Exception as e:
            print(f"환자 데이터 처리 중 에러: {str(e)}, row={row}")
            import traceback
            print(traceback.format_exc())
            raise

    print(f"환자 정보 조회 완료: {len(patients)}명")

    return PatientInfoResponse(
        patients=patients,
        total_count=len(patients),
        timestamp=timestamp
    )

async def get_patient_data_range(patient_id: int, timestamp: datetime, session: AsyncSession):
    """
    특정 환자의 특정 timestamp 기준 -8시간 ~ timestamp 사이의 데이터를 조회하여 Json으로 반환
    """
    print(f"환자 {patient_id}번 데이터 조회 시작 — 기준 timestamp = {timestamp}")

    # timezone 제거 (DB timestamp 비교용)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.replace(tzinfo=None)

    start_time = timestamp - timedelta(hours=8)
    end_time = timestamp

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

async def get_patient_predicted_by_timestamp(patient_id: int, timestamp: datetime, session: AsyncSession):
    """
    특정 환자의 기준 timestamp 이후 가장 가까운 (hour 단위로 끊은) timestamp의 news_score를 반환
    (반환 시 분·초는 잘라서 시까지만 표시)
    """
    print(f"환자 {patient_id}번 예측값 조회 시작 — 기준 timestamp = {timestamp}")

    query = text("""
        SELECT
            clinical_id,
            patient_id,
            DATE_TRUNC('hour', timestamp) AS truncated_timestamp,
            timepoint,
            news_score
        FROM public.clinical_data
        WHERE patient_id = :patient_id
          AND DATE_TRUNC('hour', timestamp) > DATE_TRUNC('hour', CAST(:timestamp AS TIMESTAMP))
        ORDER BY DATE_TRUNC('hour', timestamp) ASC
        LIMIT 1
    """)

    result = await session.execute(query, {
        "patient_id": patient_id,
        "timestamp": timestamp
    })
    row = result.fetchone()

    if not row:
        print(f"환자 {patient_id}번의 {timestamp} 이후 예측값 데이터가 없습니다.")
        return {
            "patient_id": patient_id,
            "base_timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
            "message": "해당 시점 이후의 예측값 데이터가 없습니다.",
            "data": []
        }

    # truncated_timestamp를 시간까지만 포맷
    truncated_str = row[2].strftime("%Y-%m-%d %H:%M") if row[2] else None

    data = {
        'clinical_id': row[0],
        'patient_id': row[1],
        'timestamp_hour': truncated_str,
        'timepoint': row[3],
        'news_score': int(row[4]) if row[4] is not None else None
    }

    print(f"환자 {patient_id}번 {truncated_str} (시간 단위) 예측값 조회 완료")

    return {
        "patient_id": patient_id,
        "base_timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
        "nearest_future_timestamp_hour": truncated_str,
        "data": data
    }


async def get_all_clinical_data(session: AsyncSession):
    """
    clinical_data 테이블 전체 조회하여 DataFrame으로 반환
    """
    print("전체 clinical_data 조회 시작")

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

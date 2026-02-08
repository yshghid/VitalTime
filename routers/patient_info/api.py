from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from core.database import get_db_session
from . import crud, models, ml

router = APIRouter()

@router.get("/api/get-patient-info", response_model=models.PatientInfoResponse, tags=["Patient"])
async def get_patient_info(
    timestamp: str = Query(..., description="기준 timestamp (ISO 형식 예: 2025-01-01T08:00:00)"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        # Parse ISO string to datetime
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        print(f"Parsed datetime: {dt}")
        result = await crud.get_patient_info(dt, session)
        print(f"Query result: {result}")
        return result
    except ValueError as e:
        print(f"ValueError: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Invalid timestamp format: {str(e)}")
    except Exception as e:
        import traceback
        print(f"Exception occurred: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/get-patient-data-range/{patient_id}", tags=["Clinical Data"])
async def get_patient_data_range(
        patient_id: int,
        timestamp: datetime = Query(..., description="기준 timestamp (ISO 형식 예: 2025-01-01T08:00:00)"),
        session: AsyncSession = Depends(get_db_session)
):
    try:
        return await crud.get_patient_data_range(patient_id, timestamp, session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/get-patient-predicted/{patient_id}", tags=["Clinical Data"])
async def get_patient_predicted_by_timestamp(
    patient_id: int,
    timestamp: datetime = Query(..., description="기준 timestamp (ISO 형식 예: 2025-01-01T08:00:00)"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        return await crud.get_patient_predicted_by_timestamp(patient_id, timestamp, session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/train-model", tags=["ML"])
async def train_model(session: AsyncSession = Depends(get_db_session)):
    try:
        return await ml.train_lstm_model(session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

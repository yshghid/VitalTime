from pydantic import BaseModel
from datetime import datetime
from typing import List

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

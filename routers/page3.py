
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter, FastAPI, HTTPException, Depends
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import text

from core.database import get_db_session
import os

router = APIRouter()

# 환경 변수 설정
os.environ.setdefault("OPENAI_API_KEY", "your-openai-api-key-here")

# FastAPI 앱 생성
app = FastAPI(
    title="VitalTime 3번째 페이지 API",
    description="환자/병원 정보 표시 및 AI 보고서 생성",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PostgreSQL 연결 설정
DATABASE_URL = "postgresql+asyncpg://username:password@localhost/database_name"
engine = create_async_engine(DATABASE_URL, future=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# OpenAI LLM 설정
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.3,
    max_tokens=1000
)

# =========================
# Pydantic 모델 정의
# =========================

class HospitalInfo(BaseModel):
    """병원 정보 모델"""
    id: int
    name: str
    address: str
    distance: float
    phone: str

class PatientInfo(BaseModel):
    """환자 정보 모델"""
    patient_id: int
    patient_name: str
    severity: int

class ClinicalData(BaseModel):
    """임상 검사 수치 모델"""
    d_dimer: Optional[float] = None
    ldh: Optional[float] = None
    creatinine: Optional[float] = None
    hemoglobin: Optional[float] = None
    lymphocytes: Optional[float] = None
    neutrophils: Optional[float] = None
    hs_crp: Optional[float] = None
    timepoint: int

class Page3Request(BaseModel):
    """3번째 페이지 요청 모델"""
    patient_id: int = Field(..., description="환자 ID")
    hospital_info: HospitalInfo = Field(..., description="선택된 병원 정보")

class AIReport(BaseModel):
    """AI 생성 보고서 모델"""
    report_content: str
    generated_at: datetime = Field(default_factory=datetime.now)

class Page3Response(BaseModel):
    """3번째 페이지 응답 모델"""
    patient_info: PatientInfo
    hospital_info: HospitalInfo
    clinical_data: ClinicalData
    ai_report: AIReport

# =========================
# 데이터 조회 함수들
# =========================

async def get_patient_info(patient_id: int, db: AsyncSession) -> PatientInfo:
    """환자 정보 조회"""
    query = text("""
        SELECT patient_id, patient_name, severity
        FROM patient
        WHERE patient_id = :patient_id
    """)

    result = await db.execute(query, {"patient_id": patient_id})
    row = result.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="환자 정보를 찾을 수 없습니다.")

    return PatientInfo(
        patient_id=row.patient_id,
        patient_name=row.patient_name,
        severity=row.severity
    )

async def get_latest_clinical_data(patient_id: int, db: AsyncSession) -> ClinicalData:
    """최신 임상 검사 수치 조회 (timepoint가 가장 높은 값)"""
    import random

    query = text("""
        SELECT d_dimer, ldh, creatinine, timepoint
        FROM clinical_data
        WHERE patient_id = :patient_id
        ORDER BY timepoint DESC
        LIMIT 1
    """)

    result = await db.execute(query, {"patient_id": patient_id})
    row = result.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="임상 검사 데이터를 찾을 수 없습니다.")

    return ClinicalData(
        d_dimer=row.d_dimer,
        ldh=row.ldh,
        creatinine=row.creatinine,
        hemoglobin=round(random.uniform(12.0, 16.0), 1),
        lymphocytes=round(random.uniform(20.0, 40.0), 1),
        neutrophils=round(random.uniform(40.0, 75.0), 1),
        hs_crp=round(random.uniform(0.5, 5.0), 2),
        timepoint=row.timepoint
    )

# =========================
# AI 보고서 생성 함수
# =========================

def generate_medical_report(
    patient_info: PatientInfo,
    hospital_info: HospitalInfo,
    clinical_data: ClinicalData
) -> str:
    """의료 전원 보고서 생성"""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.3,
        max_tokens=1000
    )

    prompt_template = f"""
다음 정보를 바탕으로 의료기관 간 환자 전원 의뢰서를 공식 문서 형식으로 작성해주세요.

【환자 정보】
- 환자명: {patient_info.patient_name}
- 환자 ID: {patient_info.patient_id}
- 중증도: {patient_info.severity}

【현재 입원 의료기관】
- 의료기관명: SKALA대학병원
- 담당의: {patient_info.doctor_name if hasattr(patient_info, 'doctor_name') else '미정'}
- 연락처: 02-1234-5678

【이송 예정 의료기관】
- 의료기관명: {hospital_info.name}
- 주소: {hospital_info.address}
- 연락처: {hospital_info.phone}
- 이송 거리: {hospital_info.distance}km

【최신 검사 수치 (검사일시: {datetime.now().strftime('%Y-%m-%d %H:%M')})】
- D-Dimer: {clinical_data.d_dimer if clinical_data.d_dimer else 'N/A'} ng/mL
- LDH: {clinical_data.ldh if clinical_data.ldh else 'N/A'} U/L
- Creatinine: {clinical_data.creatinine if clinical_data.creatinine else 'N/A'} mg/dL
- Hemoglobin: {clinical_data.hemoglobin if clinical_data.hemoglobin else 'N/A'} g/dL
- Lymphocytes: {clinical_data.lymphocytes if clinical_data.lymphocytes else 'N/A'}%
- Neutrophils: {clinical_data.neutrophils if clinical_data.neutrophils else 'N/A'}%
- hs-CRP: {clinical_data.hs_crp if clinical_data.hs_crp else 'N/A'} mg/L

다음 형식으로 작성해주세요:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    환자 전원 의뢰서
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

■ 환자 기본 정보
 - 성명: {patient_info.patient_name}
 - 환자번호: {patient_info.patient_id}
 - 중증도 분류: {patient_info.severity}
 - 주 증상/진단명: (검사 수치를 바탕으로 추정되는 주요 증상이나 의심 진단 작성)

■ 현재 입원 의료기관
 - 기관명: SKALA대학병원
 - 담당의: {patient_info.doctor_name if hasattr(patient_info, 'doctor_name') else '미정'}
 - 입원일자: {datetime.now().strftime('%Y년 %m월 %d일')}

■ 이송 예정 의료기관
 - 기관명: {hospital_info.name}
 - 소재지: {hospital_info.address}
 - 연락처: {hospital_info.phone}
 - 예상 이송시간: 약 {int(hospital_info.distance / 40 * 60)}분 (거리: {hospital_info.distance}km)

■ 현재 상태 및 검사 소견
 - 주요 검사 결과 (검사일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}):
   - D-Dimer: {clinical_data.d_dimer if clinical_data.d_dimer else 'N/A'} ng/mL
   - LDH: {clinical_data.ldh if clinical_data.ldh else 'N/A'} U/L
   - Creatinine: {clinical_data.creatinine if clinical_data.creatinine else 'N/A'} mg/dL
   - Hemoglobin: {clinical_data.hemoglobin if clinical_data.hemoglobin else 'N/A'} g/dL
   - Lymphocytes: {clinical_data.lymphocytes if clinical_data.lymphocytes else 'N/A'}%
   - Neutrophils: {clinical_data.neutrophils if clinical_data.neutrophils else 'N/A'}%
   - hs-CRP: {clinical_data.hs_crp if clinical_data.hs_crp else 'N/A'} mg/L

 - 활력징후:
   (혈압, 맥박, 호흡수, 체온 등을 가정하여 작성)

■ 전원 사유 및 임상적 판단
위 검사 수치를 종합적으로 분석하여:
1. 환자의 현재 상태를 의학적으로 평가
2. 전원이 필요한 구체적 사유 (예: 고위험군 치료, 전문 처치 필요 등)
3. 수용 의료기관에서 필요한 추가 검사 및 치료 계획

를 3-4문장으로 전문적으로 서술해주세요.

■ 특이사항 및 주의사항
- 이송 중 주의사항: (예: 활력징후 모니터링 필요, 산소 공급 등)
- 알레르기 정보: 없음 (또는 있다면 명시)
- 현재 투약 중인 약물: (있다면 명시)
- 기타 전달사항: (이송 시 특별히 주의할 사항)

■ 전원 일정
 - 전원 희망일시: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}
 - 이송 수단: 구급차 (또는 적절한 이송 수단)
 - 보호자 동행 여부: 동행 예정

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                                    의뢰 의료기관: SKALA대학병원
                                    담당 의사: {patient_info.doctor_name if hasattr(patient_info, 'doctor_name') else '___________'} (인)
                                    
                                    작성일시: {datetime.now().strftime('%Y. %m. %d. %H:%M:%S')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

※ 본 의뢰서는 환자의 적절한 치료를 위해 작성된 공식 의료 문서입니다.
※ 수용 의료기관에서는 환자 도착 즉시 응급처치가 가능하도록 준비 바랍니다.

의료진이 실제로 사용할 수 있을 정도로 전문적이고 정확한 내용으로 작성해주세요.
각 섹션은 의학적 근거에 기반하여 작성하되, 자연스럽고 읽기 쉬운 문장으로 구성해주세요.
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt_template)])
        return response.content.strip()
    except Exception as e:
        return f"보고서 생성 중 오류가 발생했습니다: {str(e)}"

# =========================
# API 엔드포인트
# =========================

@router.post("/api/page3/patient-report", response_model=Page3Response)
async def get_patient_report(
    request: Page3Request,
    db: AsyncSession = Depends(get_db_session)
):
    """
    3번째 페이지 - 환자/병원 정보 및 AI 보고서 조회
    """
    print(f"=== page3 API 요청 받음 ===")
    print(f"request dict: {request.dict()}")
    print(f"patient_id: {request.patient_id}")
    print(f"hospital_info: {request.hospital_info}")
    try:
        patient_info = await get_patient_info(request.patient_id, db)
        clinical_data = await get_latest_clinical_data(request.patient_id, db)
        report_content = generate_medical_report(
            patient_info=patient_info,
            hospital_info=request.hospital_info,
            clinical_data=clinical_data
        )
        ai_report = AIReport(report_content=report_content)
        return Page3Response(
            patient_info=patient_info,
            hospital_info=request.hospital_info,
            clinical_data=clinical_data,
            ai_report=ai_report
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

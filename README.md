# VitalTime
AI 기반 환자 전원 의뢰 시스템

## 0. 환경 설정

### .env 파일 생성
프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 입력합니다.
```
DATABASE_URL=postgresql+asyncpg://myuser:mypassword@localhost:5432/mydatabase
```

### config.js 생성
`config.js.example`을 복사하여 `config.js`를 생성하고, Google Maps API 키를 입력합니다.
```bash
cp config.js.example config.js
```

### DB 설정
PostgreSQL이 실행 중이어야 합니다.

```bash
# 테이블 생성 및 샘플 데이터 삽입
PGPASSWORD=mypassword psql -h localhost -p 5432 -U myuser -d mydatabase -f dummy/sample.sql
```

## 1. 백엔드 서버 실행

```bash
pip install -r requirements.txt

python main_api.py
```
또는 uvicorn으로 직접 실행:
```bash
uvicorn main_api:app --port 8001 --reload
```
백엔드 서버가 http://localhost:8001 에서 실행됩니다.

## 2. 프론트엔드 서버 실행

```bash
npm install

npm run dev
```
프론트엔드가 http://localhost:3000 에서 실행됩니다.

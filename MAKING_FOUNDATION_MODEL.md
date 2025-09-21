# 파운데이션 모델 학습 가이드

## 1. 학습 데이터 전처리

### 세법 데이터 준비
```bash
# 세법 원본 데이터를 instruction 형태로 변환
python training-data/tax-law/prepare.py
```

- **원본 데이터**: `training-data/tax-law/` 폴더의 JSON 파일들
  - 개별소비세, 교육세, 교통ㆍ에너지ㆍ환경세, 농어촌특별세, 법인세
  - 부가가치세, 상속세와 증여세, 소득세, 인지세, 종합부동산세, 주세, 증권거래세
- **처리 과정**: JSON → 조문 추출 → instruction-response 쌍 생성
- **출력**: `training-data/tax-law-instruction/`에 `train.bin`, `val.bin`, `meta.pkl` 생성

## 2. 파운데이션 모델 학습

### 학습 실행
```bash
# 세법 instruction 모델 학습
python train.py config/train_tax_law.py

# MLflow 서버 실행 (선택사항)
python start_mlflow_server.py
```

### 주요 설정
- **모델 크기**: 12 layers, 12 heads, 768 embedding
- **배치 크기**: 12 (gradient accumulation 8)
- **학습률**: 0.0006 (warmup 2000 steps)
- **최대 이터레이션**: 200,000
- **블록 크기**: 1024

## 3. 검증 방법

### 학습 중 검증
- 2000 스텝마다 자동 검증
- 200 이터레이션으로 검증 loss 계산
- MLflow로 실시간 모니터링

### 모델 평가
```bash
# 세법 질문에 대한 답변 테스트
python sample.py --out_dir=out-tax-law --start="소득세 제1조는 무엇에 관한 조문인가요?"
```

### 체크포인트
- `always_save_checkpoint=True`로 자동 저장
- `out-tax-law/ckpt.pt`에 모델 저장

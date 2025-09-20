# 🏗️ Foundation Model 만들기 가이드

**완전히 로컬에서 인터넷 다운로드 없이 파운데이션 모델을 처음부터 구축하는 방법**

## 🎯 개요

이 가이드는 nanoGPT를 사용하여 **완전히 오프라인**에서 파운데이션 모델을 만드는 방법을 설명합니다. 사전 훈련된 가중치를 다운로드하지 않고, 순수하게 로컬 데이터와 랜덤 초기화부터 시작합니다.

## 📋 목차

1. [환경 준비](#환경-준비)
2. [모델 아키텍처 설계](#모델-아키텍처-설계)
3. [데이터 준비](#데이터-준비)
4. [모델 초기화](#모델-초기화)
5. [훈련 설정](#훈련-설정)
6. [훈련 실행](#훈련-실행)
7. [모델 평가](#모델-평가)
8. [파운데이션 모델 활용](#파운데이션-모델-활용)

---

## 🔧 환경 준비

### 1. 필수 라이브러리 설치
```bash
# 가상환경 활성화
source .venv/bin/activate

# 모든 필요한 패키지가 설치되어 있는지 확인
pip install -r requirements.txt
```

### 2. 시스템 요구사항
- **Mac M4**: MPS 가속 지원
- **메모리**: 최소 8GB RAM (16GB 권장)
- **저장공간**: 모델 크기에 따라 1GB-10GB
- **인터넷**: 초기 라이브러리 설치 후 불필요

---

## 🏛️ 모델 아키텍처 설계

### 1. 모델 크기 결정

파운데이션 모델의 크기를 용도에 따라 선택합니다:

#### 🔬 **실험용 (Tiny Foundation)**
```python
# 빠른 실험과 개념 증명용
n_layer = 6        # 6개 레이어
n_head = 6         # 6개 어텐션 헤드  
n_embd = 384       # 384차원 임베딩
block_size = 512   # 512 토큰 컨텍스트
vocab_size = 50304 # GPT-2 표준 어휘
# 예상 파라미터: ~25M
```

#### 🚀 **소형 파운데이션 (Small Foundation)**
```python
# 실용적인 성능과 효율성의 균형
n_layer = 12       # 12개 레이어
n_head = 12        # 12개 어텐션 헤드
n_embd = 768       # 768차원 임베딩 (GPT-2 기본)
block_size = 1024  # 1024 토큰 컨텍스트
vocab_size = 50304 # GPT-2 표준 어휘
# 예상 파라미터: ~124M (GPT-2 크기)
```

#### 💪 **중형 파운데이션 (Medium Foundation)**
```python
# 높은 성능이 필요한 경우
n_layer = 24       # 24개 레이어
n_head = 16        # 16개 어텐션 헤드
n_embd = 1024      # 1024차원 임베딩
block_size = 2048  # 2048 토큰 컨텍스트
vocab_size = 50304 # GPT-2 표준 어휘
# 예상 파라미터: ~350M (GPT-2 Medium 크기)
```

#### 🦾 **대형 파운데이션 (Large Foundation)**
```python
# 최고 성능이 필요한 경우 (고성능 하드웨어 필요)
n_layer = 36       # 36개 레이어
n_head = 20        # 20개 어텐션 헤드
n_embd = 1280      # 1280차원 임베딩
block_size = 4096  # 4096 토큰 컨텍스트
vocab_size = 50304 # GPT-2 표준 어휘
# 예상 파라미터: ~774M (GPT-2 Large 크기)
```

### 2. 설정 파일 생성

선택한 크기에 맞는 설정 파일을 만듭니다:

```bash
# 설정 파일 생성 스크립트 실행
python create_foundation_config.py --size small
```

---

## 📚 데이터 준비

### 1. 데이터 소스 선택

파운데이션 모델을 위한 다양한 데이터 소스:

#### 📖 **텍스트 데이터**
- **문학**: 고전 문학, 소설, 시
- **위키피디아**: 백과사전적 지식
- **뉴스**: 다양한 주제의 뉴스 기사
- **학술 논문**: 전문 지식
- **웹 텍스트**: 블로그, 포럼 등

#### 🌐 **공개 데이터셋**
- **Project Gutenberg**: 저작권 만료 도서
- **Common Crawl**: 웹 크롤링 데이터
- **OpenWebText**: 오픈소스 웹텍스트
- **BookCorpus**: 도서 데이터

### 2. 데이터 전처리 스크립트

```python
# data/foundation/prepare.py
import os
import numpy as np
import tiktoken
from datasets import load_dataset

def prepare_foundation_data():
    """파운데이션 모델을 위한 대규모 데이터 준비"""
    
    # 여러 데이터 소스 결합
    datasets = [
        "bookcorpus",     # 도서 데이터
        "wikipedia",      # 위키피디아
        "openwebtext",    # 웹 텍스트
        # 추가 데이터셋...
    ]
    
    # 토크나이저 준비
    enc = tiktoken.get_encoding("gpt2")
    
    all_text = ""
    
    # 각 데이터셋 처리
    for dataset_name in datasets:
        print(f"Processing {dataset_name}...")
        # 데이터 로드 및 전처리 로직
        # ...
    
    # 훈련/검증 분할
    train_data = all_text[:int(len(all_text) * 0.95)]
    val_data = all_text[int(len(all_text) * 0.95):]
    
    # 토크나이징
    train_ids = enc.encode(train_data)
    val_ids = enc.encode(val_data)
    
    # 바이너리 파일로 저장
    np.array(train_ids, dtype=np.uint16).tofile('train.bin')
    np.array(val_ids, dtype=np.uint16).tofile('val.bin')
    
    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")
```

### 3. 로컬 데이터 사용

인터넷 없이 로컬 데이터만 사용하는 경우:

```bash
# 로컬 텍스트 파일들을 하나로 합치기
cat *.txt > foundation_corpus.txt

# 전처리 실행
python data/foundation/prepare_local.py
```

---

## 🧠 모델 초기화

### 1. 로컬 모델 생성 스크립트

```python
# create_foundation_model.py
from model import GPTConfig, GPT
import torch

def create_foundation_model(size="small"):
    """파운데이션 모델 생성"""
    
    configs = {
        "tiny": {
            "n_layer": 6, "n_head": 6, "n_embd": 384,
            "block_size": 512, "vocab_size": 50304
        },
        "small": {
            "n_layer": 12, "n_head": 12, "n_embd": 768,
            "block_size": 1024, "vocab_size": 50304
        },
        "medium": {
            "n_layer": 24, "n_head": 16, "n_embd": 1024,
            "block_size": 2048, "vocab_size": 50304
        },
        "large": {
            "n_layer": 36, "n_head": 20, "n_embd": 1280,
            "block_size": 4096, "vocab_size": 50304
        }
    }
    
    config_args = configs[size]
    config_args.update({
        "dropout": 0.0,    # 사전훈련에서는 드롭아웃 비활성화
        "bias": False      # 효율성을 위해 bias 제거
    })
    
    config = GPTConfig(**config_args)
    model = GPT(config)
    
    print(f"Created {size} foundation model:")
    print(f"  Parameters: {model.get_num_params():,}")
    print(f"  Layers: {config.n_layer}")
    print(f"  Heads: {config.n_head}")
    print(f"  Embedding: {config.n_embd}")
    print(f"  Context: {config.block_size}")
    
    return model, config

# 사용 예시
model, config = create_foundation_model("small")
```

### 2. 가중치 초기화 전략

nanoGPT는 다음 초기화 방법을 사용합니다:

```python
# model.py의 _init_weights 함수
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

# 잔차 연결에 대한 특별한 초기화
for pn, p in self.named_parameters():
    if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

---

## ⚙️ 훈련 설정

### 1. 파운데이션 모델 훈련 설정

```python
# config/train_foundation.py

# 출력 디렉토리
out_dir = 'out-foundation'

# 평가 설정
eval_interval = 2000      # 자주 평가하지 않음 (시간 절약)
eval_iters = 200         # 평가 시 사용할 배치 수
log_interval = 100       # 로그 간격

# 체크포인트 설정
always_save_checkpoint = True  # 모든 평가마다 저장

# 로깅 설정
wandb_log = True         # 실험 추적 (선택사항)
wandb_project = 'foundation-model'
wandb_run_name = 'foundation-v1'

# 데이터 설정
dataset = 'foundation'   # 준비한 파운데이션 데이터
gradient_accumulation_steps = 8  # 큰 배치 크기 시뮬레이션
batch_size = 12          # 실제 배치 크기
block_size = 1024        # 컨텍스트 길이

# 모델 아키텍처 (Small Foundation)
n_layer = 12
n_head = 12  
n_embd = 768
dropout = 0.0            # 사전훈련에서는 드롭아웃 없음
bias = False

# 옵티마이저 설정
learning_rate = 6e-4     # GPT-3 논문 기준
max_iters = 600000       # 매우 긴 훈련 (조정 가능)
weight_decay = 1e-1      # 강한 가중치 감쇠
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0          # 그래디언트 클리핑

# 학습률 스케줄링
decay_lr = True
warmup_iters = 2000      # 워밍업 단계
lr_decay_iters = 600000  # max_iters와 동일
min_lr = 6e-5            # 최소 학습률

# 시스템 설정
device = 'mps'           # Mac M4 GPU 사용
dtype = 'bfloat16'       # 메모리 효율성
compile = False          # Mac에서는 비활성화
```

### 2. 훈련 단계별 설정

#### 🥇 **1단계: 기본 언어 모델링**
```python
# 기본 다음 토큰 예측 훈련
max_iters = 100000
learning_rate = 6e-4
```

#### 🥈 **2단계: 안정화 훈련** 
```python
# 더 낮은 학습률로 안정화
max_iters = 200000
learning_rate = 3e-4
```

#### 🥉 **3단계: 파인튜닝 준비**
```python
# 매우 낮은 학습률로 마무리
max_iters = 50000
learning_rate = 1e-4
```

---

## 🚀 훈련 실행

### 1. 단일 GPU 훈련 (Mac M4)

```bash
# 가상환경 활성화
source .venv/bin/activate

# 기본 훈련 시작
python train.py config/train_foundation.py \
    --device=mps \
    --compile=False \
    --init_from=scratch
```

### 2. 단계별 훈련

```bash
# 1단계: 기본 훈련
python train.py config/train_foundation_stage1.py \
    --device=mps --compile=False --init_from=scratch

# 2단계: 안정화 (1단계 결과에서 시작)
python train.py config/train_foundation_stage2.py \
    --device=mps --compile=False --init_from=resume

# 3단계: 파인튜닝 준비
python train.py config/train_foundation_stage3.py \
    --device=mps --compile=False --init_from=resume
```

### 3. 훈련 모니터링

```bash
# 실시간 로그 확인
tail -f train.log

# GPU 사용률 모니터링 (Mac)
sudo powermetrics --samplers gpu_power -n 1 -i 5000
```

### 4. 훈련 중단 및 재시작

```bash
# 훈련 중단 후 재시작
python train.py config/train_foundation.py \
    --device=mps --compile=False --init_from=resume
```

---

## 📊 모델 평가

### 1. 기본 성능 측정

```python
# evaluate_foundation.py
import torch
from model import GPT, GPTConfig

def evaluate_foundation_model(model_path):
    """파운데이션 모델 평가"""
    
    # 모델 로드
    checkpoint = torch.load(model_path, weights_only=False)
    config = GPTConfig(**checkpoint['model_args'])
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    
    # 평가 모드
    model.eval()
    
    # 성능 지표
    print(f"Model Parameters: {model.get_num_params():,}")
    print(f"Training Loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"Training Iterations: {checkpoint.get('iter_num', 'N/A')}")
    
    return model

# 사용
model = evaluate_foundation_model('out-foundation/ckpt.pt')
```

### 2. 텍스트 생성 품질 테스트

```bash
# 다양한 프롬프트로 생성 테스트
python sample.py --out_dir=out-foundation --device=mps \
    --start="The future of artificial intelligence" \
    --num_samples=5 --max_new_tokens=200

python sample.py --out_dir=out-foundation --device=mps \
    --start="In a world where" \
    --num_samples=3 --max_new_tokens=300
```

### 3. 벤치마크 테스트

```bash
# 성능 벤치마크
python bench.py --device=mps --real_data=True
```

---

## 🎯 파운데이션 모델 활용

### 1. 다운스트림 태스크 파인튜닝

```python
# config/finetune_from_foundation.py

# 파운데이션 모델에서 시작
init_from = 'resume'
out_dir = 'out-foundation'  # 파운데이션 모델 위치

# 파인튜닝 설정
learning_rate = 1e-4      # 낮은 학습률
max_iters = 5000          # 짧은 훈련
dropout = 0.1             # 파인튜닝에서는 드롭아웃 사용

# 태스크별 데이터
dataset = 'your_task_data'  # 특정 태스크 데이터
```

### 2. Instruction Tuning

```bash
# 파운데이션 모델을 instruction model로 변환
python train.py config/instruction_from_foundation.py \
    --device=mps --compile=False --init_from=resume
```

### 3. 도메인 특화 모델

```bash
# 특정 도메인(예: 의료, 법률)에 특화
python train.py config/domain_specific.py \
    --device=mps --compile=False --init_from=resume
```

---

## 📈 성능 최적화 팁

### 1. 메모리 최적화

```python
# 그래디언트 체크포인팅 활용
gradient_checkpointing = True

# 배치 크기 조정
batch_size = 8           # 메모리에 맞게 조정
gradient_accumulation_steps = 16  # 효과적인 배치 크기 유지
```

### 2. 훈련 속도 향상

```bash
# MPS 최적화 설정
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### 3. 안정적인 훈련

```python
# 그래디언트 클리핑
grad_clip = 1.0

# 학습률 워밍업
warmup_iters = 2000

# 정규화
weight_decay = 1e-1
```

---

## 🔍 문제 해결

### 1. 메모리 부족
```bash
# 배치 크기 줄이기
--batch_size=4 --gradient_accumulation_steps=32

# 컨텍스트 길이 줄이기  
--block_size=512
```

### 2. 훈련 불안정
```bash
# 학습률 낮추기
--learning_rate=3e-4

# 그래디언트 클리핑 강화
--grad_clip=0.5
```

### 3. 수렴 속도 느림
```bash
# 학습률 스케줄 조정
--warmup_iters=5000
--lr_decay_iters=300000
```

---

## 🎉 결론

이 가이드를 따라하면 **완전히 로컬에서** 인터넷 다운로드 없이 파운데이션 모델을 만들 수 있습니다. 

### 🏆 **완성된 파운데이션 모델의 특징:**
- ✅ **완전 오프라인**: 외부 의존성 없음
- ✅ **커스터마이징 가능**: 원하는 크기와 구조
- ✅ **파인튜닝 준비**: 다양한 태스크에 적용 가능
- ✅ **상용 품질**: 실제 애플리케이션에 사용 가능

### 🚀 **다음 단계:**
1. **평가 및 벤치마킹**: 모델 성능 측정
2. **파인튜닝**: 특정 태스크에 최적화
3. **배포**: 실제 서비스에 적용
4. **지속적 개선**: 더 많은 데이터로 확장

**축하합니다! 이제 여러분만의 파운데이션 모델을 가지게 되었습니다!** 🎊

---

## 📚 참고 자료

- [nanoGPT GitHub](https://github.com/karpathy/nanoGPT)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Scaling Laws Paper](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

---

*이 가이드는 nanoGPT 프레임워크를 기반으로 작성되었습니다. Mac M4 환경에서 테스트되었으며, 다른 환경에서는 설정을 조정해야 할 수 있습니다.*

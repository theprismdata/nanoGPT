# 🤖 nanoGPT Instruction Model Testing Guide

이 가이드는 훈련된 instruction 모델을 쉽게 테스트할 수 있는 방법들을 제공합니다.

## 🚀 빠른 시작

### 1. 단일 질문 테스트
```bash
# 가상환경 활성화
source .venv/bin/activate

# 간단한 질문
python test_instruction.py --question "What is machine learning?"

# 입력이 있는 질문
python test_instruction.py --question "Translate to Korean:" --input "Hello world"

# 온도 조절 (창의성 조절)
python test_instruction.py --question "Write a poem about AI" --temperature 0.9
```

### 2. 대화형 모드
```bash
python test_instruction.py --interactive
```
- 연속적으로 질문을 입력할 수 있습니다
- `quit` 또는 `exit`으로 종료
- 각 질문마다 온도 설정 가능

### 3. 배치 테스트
```bash
python test_instruction.py --batch
```
- 미리 정의된 다양한 테스트를 순차적으로 실행
- 모델의 전반적인 성능을 확인

## 📝 명령어 옵션

| 옵션 | 단축 | 설명 | 기본값 |
|------|------|------|--------|
| `--question` | `-q` | 테스트할 질문 | - |
| `--input` | `-i` | 질문에 대한 입력 텍스트 | "" |
| `--temperature` | `-t` | 샘플링 온도 (0.1-1.0) | 0.7 |
| `--max_tokens` | `-m` | 최대 생성 토큰 수 | 200 |
| `--interactive` | - | 대화형 모드 | False |
| `--batch` | - | 배치 테스트 모드 | False |

## 🎯 테스트 예시

### 질문-답변
```bash
python test_instruction.py -q "What is the capital of France?"
```

### 번역
```bash
python test_instruction.py -q "Translate to Korean:" -i "Good morning"
```

### 코드 생성
```bash
python test_instruction.py -q "Write a Python function to calculate factorial" -t 0.4
```

### 창작
```bash
python test_instruction.py -q "Write a short story about a robot" -t 0.9 -m 300
```

### 요약
```bash
python test_instruction.py -q "Summarize this text:" -i "Long text here..."
```

## 🌡️ 온도 설정 가이드

- **0.1-0.4**: 정확한 답변이 필요한 경우 (사실, 번역, 코드)
- **0.5-0.7**: 균형 잡힌 답변 (일반적인 질문)
- **0.8-1.0**: 창의적인 답변이 필요한 경우 (시, 소설, 아이디어)

## 🔧 고급 사용법

### 원본 sample.py 직접 사용
```bash
python sample.py \
    --out_dir=out-instruction \
    --device=mps \
    --start="<|instruction|>Your question here<|response|>" \
    --num_samples=1 \
    --max_new_tokens=200 \
    --temperature=0.7
```

### 여러 샘플 생성
```bash
python test_instruction.py -q "Write a haiku about programming" -t 0.8 --max_tokens 100
```

## 🐛 문제 해결

### 모델이 없다는 오류
```
❌ Error: No instruction model found!
```
**해결법**: 먼저 instruction 모델을 훈련하세요
```bash
python train.py config/train_instruction.py --device=mps --compile=False
```

### 응답이 이상할 때
1. 온도를 낮춰보세요 (`--temperature 0.5`)
2. 질문을 더 구체적으로 작성하세요
3. 입력 형식을 확인하세요

### 한국어 깨짐 현상
- GPT-2 토크나이저의 한계로 일부 한국어가 깨질 수 있습니다
- 영어 위주의 질문으로 테스트해보세요

## 📊 성능 팁

- **Mac M4**: MPS 가속으로 빠른 추론 가능
- **메모리**: 50M 파라미터 모델로 가벼움
- **속도**: 200토큰 생성에 약 10-20초

## 🎪 재미있는 테스트 아이디어

```bash
# AI에 대한 철학적 질문
python test_instruction.py -q "What makes humans different from AI?"

# 창의적 문제 해결
python test_instruction.py -q "How would you design a house for cats?" -t 0.8

# 코딩 도우미
python test_instruction.py -q "Debug this Python code:" -i "def hello(): print('Hello' + name)"

# 언어 학습
python test_instruction.py -q "Explain the difference between 'a' and 'an' in English"

# 요리 레시피
python test_instruction.py -q "How do you make kimchi fried rice?"
```

Happy testing! 🚀

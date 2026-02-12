# Qwen3-TTS Finetune Studio

`QwenLM/Qwen3-TTS` 공식 `finetuning` 스크립트를 기반으로, 기존 `qwen3-tts-studio` UX 흐름을 살린 파인튜닝/운영 통합 툴입니다.

이 문서는 현재 코드베이스 전체를 기준으로 구조, 기능, 실행 방법, 제약사항을 한 번에 정리합니다.

## 1) 프로젝트 범위

- 공식 학습 스크립트 래핑:
  - `third_party/Qwen3-TTS/finetuning/prepare_data.py`
  - `third_party/Qwen3-TTS/finetuning/sft_12hz.py`
- UI 기반 워크플로우:
  - 데이터 업로드/정리
  - 품질 검증/사전 리스크 점검
  - 오디오 정규화
  - prepare/train/pipeline 실행 및 로그 스트리밍
  - 체크포인트 추론/배치 생성
  - 런 레지스트리/체크포인트 패키징
- CLI 기반 자동화:
  - validate/preflight/precheck/normalize/prepare/train/pipeline/infer

현재 공식 스펙 기준으로 **single-speaker fine-tuning**만 지원합니다.

## 2) 전체 아키텍처

```text
raw data (audio/text/ref_audio)
  -> train_raw.jsonl
  -> validate + preflight/precheck + normalize(optional)
  -> prepare_data.py (audio_codes 추출)
  -> train_with_codes.jsonl
  -> sft_12hz.py (SFT)
  -> checkpoint-epoch-*
  -> inference(single/batch) + export(zip)
```

레이어 구성:

- UI 레이어: `qwen_finetune_ui.py`
- CLI 레이어: `qwen_finetune_cli.py`
- 도메인 로직: `finetune_studio/*.py`
- 공식 학습 코드: `third_party/Qwen3-TTS/finetuning/*`
- 실행 산출물: `workspace/`

## 3) 코드 구조 상세

### 엔트리포인트

- `qwen_finetune_ui.py`
  - Gradio 앱 생성/이벤트 바인딩
  - 8개 탭 제공 (Dataset, Quality, Prepare, Train, Pipeline, Inference, Runs, Workspace)
  - 도메인 모듈 호출 및 상태 업데이트

- `qwen_finetune_cli.py`
  - 서브커맨드 라우팅
  - 배치/서버 자동화용 인터페이스
  - `speaker-name`은 `train/pipeline/infer`에서 필수

### 도메인 모듈

- `finetune_studio/paths.py`
  - 프로젝트/워크스페이스 경로 관리
  - `QWEN_FT_WORKSPACE` 지원
  - dataset/run/checkpoint 탐색 함수 제공

- `finetune_studio/dataset_ops.py`
  - transcript(`csv/json/jsonl`) 파싱
  - 오디오 경로 해석(절대/상대/업로드명 매칭)
  - `train_raw.jsonl` 생성, import, preview/stats

- `finetune_studio/quality.py`
  - `validate_dataset`: 데이터셋 오류/경고 리포트
  - `run_preflight_review`: 학습 전 리스크 점검
    - 장치 가능 여부(CUDA/MPS/CPU)
    - 모델 경로 유효성(local/hub)
    - 디스크 여유 추정
    - 신호 품질 샘플링(SNR/clipping/silence/DC)
    - 텍스트 다양성/중복 경고
  - 품질/프리플라이트 리포트 포맷/저장

- `finetune_studio/audio_prep.py`
  - 데이터셋 오디오 정규화
  - 24kHz mono 리샘플
  - 피크 정규화 옵션
  - 정규화된 새 `train_raw.jsonl` 생성

- `finetune_studio/process_runner.py`
  - prepare/train 하위 프로세스 실행/중단
  - 동시 실행 key 잠금 관리

- `finetune_studio/training_ops.py`
  - `run_prepare_data`: 공식 `prepare_data.py` 실행
  - `run_training`: 공식 `sft_12hz.py` 실행
  - 진행률 파싱(`Epoch | Step | Loss`)
  - `run_config.json`, `train.log`, `run_summary.json` 기록

- `finetune_studio/pipeline_ops.py`
  - prepare + train 연결 실행
  - stage별 이벤트 반환

- `finetune_studio/inference_ops.py`
  - 체크포인트 로드/캐시
  - single/batch 생성
  - 배치 메타데이터 + zip 생성
  - 모델 캐시 해제

- `finetune_studio/run_registry.py`
  - `run_summary.json` 업데이트/조회
  - 런 테이블 데이터 생성

- `finetune_studio/export_ops.py`
  - 체크포인트 패키징(zip)
  - manifest/run_config/run_summary 포함
  - optimizer/scheduler/rng 파일 포함 여부 옵션

### 공식 코드 복사본

- `third_party/Qwen3-TTS/finetuning/prepare_data.py`
  - `audio_codes` 추출

- `third_party/Qwen3-TTS/finetuning/sft_12hz.py`
  - SFT 학습 수행
  - 체크포인트 산출

- `third_party/Qwen3-TTS/finetuning/dataset.py`
  - 학습용 dataset/collate 구현

## 4) 데이터 계약 (Data Contract)

### 입력(JSONL)

각 줄 최소 키:

- `audio`: 학습 대상 wav 경로
- `text`: 전사 텍스트
- `ref_audio`: 화자 참조 wav 경로

예시:

```jsonl
{"audio":"/abs/path/utt0001.wav","text":"안녕하세요.","ref_audio":"/abs/path/ref.wav"}
{"audio":"/abs/path/utt0002.wav","text":"오늘 날씨가 좋네요.","ref_audio":"/abs/path/ref.wav"}
```

권장:

- `ref_audio`는 전 샘플에서 동일 파일 사용
- audio/ref_audio 모두 24kHz mono
- 짧은 노이즈/클리핑 구간 최소화

### prepare 이후(JSONL)

- `audio_codes` 필드가 추가된 `train_with_codes.jsonl`

### 학습 전 필수/권장 요구사항 (Go/No-Go)

- 필수(하나라도 실패하면 `NO-GO`):
  - dataset `ERROR` 개수 `0`
  - init model path 유효
  - 선택 device 사용 가능
  - 디스크 여유량 >= 예상 필요량
- 권장(실패 시 `GO-WITH-CAUTION`):
  - 총 음성 길이 `>= 10분`
  - `ref_audio`는 단일 파일 유지
  - train audio 24kHz mono
  - 텍스트 다양성 높게 유지(중복 전사 최소화)
  - 클리핑/저 SNR 비율 낮게 유지

학습 시작 전 `preflight` 또는 `precheck`를 반드시 실행해 `GO/GO-WITH-CAUTION/NO-GO` 판정을 확인하세요.

## 5) UI 탭별 동작 요약

1. `1) Dataset`
- 업로드 데이터셋 구성
- 기존 raw jsonl import
- preview/statistics 확인

2. `2) Quality & Normalize`
- 품질 검증 리포트(JSON)
- 추천 학습 하이퍼파라미터 자동 반영
- Preflight Go/No-Go(READY/CAUTION/BLOCKED + 요구사항 Pass/Fail + 리소스 추정)
- Normalize(24k mono + peak normalize)

3. `3) Prepare Codes`
- `prepare_data.py` 실행/로그/중단
- output jsonl 선택 갱신

4. `4) Train`
- `sft_12hz.py` 실행/진행률/로그/중단
- run/checkpoint 자동 인덱싱
- (Safety Gate) 기본값으로 preflight `NO-GO/BLOCKED`면 실행 차단(필요시 해제 가능)

5. `5) Full Pipeline`
- prepare+train 일괄 실행
- stage별 진행 상태 출력
- (Safety Gate) 기본값으로 preflight `NO-GO/BLOCKED`면 실행 차단(필요시 해제 가능)

6. `6) Inference`
- single/batch 음성 생성
- decoding 파라미터 조정
- Quick Presets(Fast/Balanced/Quality) + 자동 저장(`workspace/ui_settings.json`)
- 체크포인트 선택 시 `run_summary.json` 기반 speaker name 자동 채움(가능한 경우)
- 체크포인트 경로는 워크스페이스 체크포인트 외에 로컬 경로/HF repo id 직접 입력도 가능
- batch zip 다운로드

7. `7) Runs & Export`
- run registry 테이블
- run summary 조회
- checkpoint package zip export

8. `8) Workspace`
- 산출물 개요
- 환경 체크(모듈/스크립트)

## 6) CLI 레퍼런스

```bash
python3 qwen_finetune_cli.py validate --raw-jsonl /abs/path/train_raw.jsonl --output-report /tmp/validate.json
python3 qwen_finetune_cli.py preflight --raw-jsonl /abs/path/train_raw.jsonl --init-model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base --prepare-device auto --batch-size 2 --num-epochs 8 --output-report /tmp/preflight.json
python3 qwen_finetune_cli.py precheck --raw-jsonl /abs/path/train_raw.jsonl --init-model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base --prepare-device auto --batch-size 2 --num-epochs 8 --output-report /tmp/precheck.json
python3 qwen_finetune_cli.py normalize --raw-jsonl /abs/path/train_raw.jsonl --name my_norm --target-sr 24000 --peak-normalize
python3 qwen_finetune_cli.py prepare --input-jsonl /abs/path/train_raw.jsonl --device auto --tokenizer-model-path Qwen/Qwen3-TTS-Tokenizer-12Hz --output-filename train_with_codes.jsonl --batch-infer-num 32
python3 qwen_finetune_cli.py train --train-jsonl /abs/path/train_with_codes.jsonl --speaker-name my_speaker --run-name run_a --batch-size 2 --learning-rate 2e-5 --num-epochs 3 --attn-implementation auto --max-steps 0
python3 qwen_finetune_cli.py pipeline --raw-jsonl /abs/path/train_raw.jsonl --prepare-device auto --speaker-name my_speaker --run-name run_pipeline --batch-size 2 --learning-rate 2e-5 --num-epochs 3 --prepare-batch-infer-num 32 --attn-implementation auto
python3 qwen_finetune_cli.py infer --checkpoint /abs/path/checkpoint-epoch-2 --speaker-name my_speaker --language auto --instruct "calm style" --text "안녕하세요" --device auto
```

exit code 규칙:

- `validate`: 데이터 오류 있으면 `2`
- `preflight`: `NO-GO/BLOCKED`면 `2`
- `precheck`: `NO-GO/BLOCKED`면 `2`
- 그 외 실행 실패는 `1`

## 7) 파라미터 제어 범위

### 이 툴에서 직접 조정 가능한 항목

- prepare:
  - `device`
  - `tokenizer_model_path`
  - `input_jsonl/output_jsonl`
  - `batch_infer_num` (오디오 코드 추출 배치 크기)

- train:
  - `init_model_path`
    - HuggingFace repo id 또는 로컬 디렉토리 모두 지원(필요 시 학습 시점에 snapshot 다운로드)
  - `train_jsonl`
  - `run_name`
  - `batch_size`
  - `learning_rate`
  - `num_epochs`
  - `speaker_name` (필수)
  - (advanced)
    - `speaker_id`
    - `gradient_accumulation_steps`
    - `mixed_precision`
    - `torch_dtype`
    - `attn_implementation`
    - `weight_decay`
    - `max_grad_norm`
    - `subtalker_loss_weight`
    - `log_every_n_steps`
    - `save_every_n_epochs`
    - `max_steps` (스모크 테스트/짧은 러닝용)
    - `seed`

- infer:
  - `device`
  - `speaker_name`
  - `language` (`auto` 포함)
  - `instruct` (스타일/톤 제어 텍스트)
  - `temperature`
  - `top_k`
  - `top_p`
  - `repetition_penalty`
  - `max_new_tokens`
  - `subtalker_temperature`
  - `subtalker_top_k`
  - `subtalker_top_p`

### 공식 스크립트 변경점(본 스튜디오에서 인자 노출)

본 프로젝트는 “공식 로직을 그대로 쓰되 운영 가능한 수준의 파라미터 제어”를 위해,
공식 스크립트에 **하위 호환 유지 범위 내에서 인자만 추가로 노출**했습니다.

- `third_party/Qwen3-TTS/finetuning/prepare_data.py`
  - `--batch_infer_num`
- `third_party/Qwen3-TTS/finetuning/sft_12hz.py`
  - `--speaker_id`
  - `--gradient_accumulation_steps`
  - `--mixed_precision`
  - `--torch_dtype`
  - `--attn_implementation`
  - `--weight_decay`
  - `--max_grad_norm`
  - `--subtalker_loss_weight`
  - `--log_every_n_steps`
  - `--save_every_n_epochs`
  - `--max_steps`
  - `--seed`

### 아직 고정/제약인 부분

- 공식 구현이 단일 화자(single-speaker)만 지원(현재 스튜디오도 동일)
- 학습 루프/모델 구조 자체는 공식 구현에 고정(optimizer 종류, 입력 임베딩 구성 등)
- 체크포인트 저장 형식은 `checkpoint-epoch-*` 디렉토리 생성 방식 유지

## 8) Preflight 판정 기준

- `READY`: blocker 없음, 주의사항 없거나 경미
- `CAUTION`: 학습은 가능하나 리스크 존재
- `BLOCKED`: 학습 전에 해결 필수 이슈 존재

주요 점검:

- 데이터 오류 존재 여부
- 장치 가용성(CUDA/MPS/CPU)
- 모델 경로 유효성
- 디스크 여유량 추정
- 신호 품질(노이즈/클리핑/무음/DC)
- 텍스트 다양성/중복
- 배치/에폭 과적합 위험 휴리스틱

## 9) 워크스페이스 산출물

```text
workspace/
├── datasets/
│   └── <dataset_name>/
│       ├── train_raw.jsonl
│       ├── source_transcript.*
│       ├── source_train_raw.jsonl
│       ├── quality_report_*.json
│       ├── preflight_report_*.json
│       ├── prepare_data.log
│       ├── train_with_codes*.jsonl
│       └── normalize_meta.json
├── runs/
│   └── <run_name>/
│       ├── run_config.json
│       ├── run_summary.json
│       ├── train.log
│       └── checkpoint-epoch-*/
└── exports/
    ├── single/single_*.wav
    ├── batch_*/wav/*.wav
    ├── batch_*.zip
    └── checkpoint_package_*/<checkpoint>.zip
```

## 10) 설치 및 실행

```bash
cd /path/to/qwen3-tts-finetune-studio
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

UI:

```bash
python3 qwen_finetune_ui.py
```

Makefile:

```bash
make setup
make ui
make check
```

환경변수:

- `QWEN_FT_WORKSPACE` (기본 `./workspace`)
- `GRADIO_SERVER_NAME` (기본 `127.0.0.1`)
- `GRADIO_SERVER_PORT` (기본 `7861`)

## 11) 운영 권장사항

- CUDA 환경을 우선 사용
- `ref_audio`는 3~10초 고품질 단일 파일 권장
- normalize 후 validate/preflight 재실행 권장
- 작은 데이터셋은 과적합 모니터링(중간 체크포인트 청취)
- 디스크는 예상 필요량 대비 30% 이상 여유 권장

## 12) 현재 제약사항

- 공식 스펙상 multi-speaker fine-tuning 미지원
- `1.7B` 베이스 모델 학습은 GPU 환경 권장(다운로드/메모리/시간 이슈로 CPU/MPS에서는 실패하거나 매우 느릴 수 있음)
- objective eval(CER/WER/MCD) 자동 측정 미구현
- tensorboard 시각화 패널 미구현
- early stopping/자동 best-checkpoint 선택 미구현

## 13) 참고 문서

- 기능 요약: `FEATURE_MATRIX.md`
- 공식 파인튜닝 안내(복사본): `third_party/Qwen3-TTS/finetuning/README.md`

## 14) E2E 스모크 테스트

이 저장소는 “전체 파이프라인이 실제로 동작한다”를 빠르게 검증하기 위해 E2E 스모크 테스트 스크립트를 제공합니다.

```bash
cd /path/to/qwen3-tts-finetune-studio
.venv/bin/python scripts/e2e_smoke.py
```

설명:

- 작은 합성(sine) 오디오로 dataset → validate → preflight → prepare → train(max_steps=1) → inference까지 한 번에 실행합니다.
- 기본값은 학습 안정성 확인용이며(학습률 `E2E_LR=0.0`), 품질 평가 목적이 아닙니다.
- 기본적으로 생성된 dataset/run 산출물은 정리(cleanup)합니다. `KEEP_E2E_ARTIFACTS=1`이면 남겨둡니다.

주요 환경변수:

- 모델 경로(로컬 우선):
  - `QWEN3_TTS_MODELS_ROOT`: 로컬 모델 루트(예: `.../qwen3-tts-studio/qwen3-TTS-studio`)
  - `E2E_INIT_MODEL_PATH`: init model 로컬 경로 또는 HF repo id (기본: 0.6B Base)
  - `E2E_TOKENIZER_MODEL_PATH`: tokenizer 로컬 경로 또는 HF repo id
  - `E2E_ALLOW_HF_DOWNLOAD=1`: 로컬 경로를 못 찾을 때 HuggingFace 다운로드 허용
- 실행 제어:
  - `E2E_PREPARE_DEVICE`: `auto|cuda:0|mps|cpu` (기본 `auto`)
  - `E2E_INFER_DEVICE`: `auto|cuda:0|mps|cpu` (기본은 CUDA 우선, 그 외 CPU)
  - `E2E_LR`: 학습률(기본 `0.0`)
  - `KEEP_E2E_ARTIFACTS=1`: cleanup 생략

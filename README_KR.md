# 유전적 친족 관계 분류기 (Genomic Relatedness Classifier)

IBD 지표와 분포 통계(feature)로 친족 관계를 예측하는 모델을 구축하고 평가합니다. 데이터셋(cM_1, cM_3, cM_6)별로 하나의 통합 리포트를 생성합니다.

## 프로젝트 구조

- `data/raw/` — 입력 원본 데이터
  - `model_input_with_kinship_filtered_cM_*.csv` — cM 임계값(1, 3, 6)별 라벨 데이터
  - `merged_info.out` (및 `merged_info.out.zip`) — 분포 통계 원본

- `data/processed/` — 전처리/중간 산출물
  - `merged_cM_*.csv` — 모델 입력용 병합 데이터
  - `top_features_*.pkl`, `scaler_*.pkl` — 특성 선택/스케일러
  - `evaluation_results_*_*.json` — 모드별 평가 JSON (통합 후에는 참조용)

- `models/<dataset>/<mode>/` — 학습된 모델 가중치(`mlp.pth`, `cnn.pth`) — git 무시

- `reports/<dataset>/` — 통합 리포트와 플롯 — git 무시
  - `results.json`, `results.md`, `results.pdf`
  - `results_KR.md`, `results_KR.pdf`
  - `feature_importance_<dataset>.png`, `kinship_distribution_<dataset>.png`
  - `<mode>/confusion_matrix_*.png` — 모드/모델별 혼동 행렬 이미지

- `scripts/` — 파이프라인 스크립트
  - `run_all.py`, `data_prep.py`, `eda.py`, `feature_selection.py`
  - `train_models.py`, `evaluate_models.py`, `build_report.py`

- `docs/` — 문서 (`plan.md`, `plan_KR.md`)

## 준비 (Windows / PowerShell)

필수 조건:
- Python 3.10–3.12
- NVIDIA GPU 및 호환 CUDA 드라이버

단계:
1) (선택) uv 설치: `pip install uv`
2) 가상환경 생성/활성화: `uv venv` → `.\.venv\Scripts\activate`
3) PyTorch(CUDA) 설치: 공식 셀렉터로 환경에 맞게 설치
   - 예: CUDA 12.1 → `pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`
4) 필수 라이브러리: `pip install pandas scikit-learn imbalanced-learn matplotlib seaborn`
5) (선택) PDF 출력용 Node.js 설치 (npx md-to-pdf)

## 사용법

데이터셋 단일 실행:
```powershell
.\.venv\Scripts\python.exe scripts\run_all.py cM_1 --epochs 1 --train-device cuda --eval-device cuda
```

모든 데이터셋 실행:
```powershell
.\.venv\Scripts\python.exe scripts\run_all.py all --epochs 1 --train-device cuda --eval-device cuda
```

옵션:
- `--epochs <int>`: 에포크 수 (기본값 1 또는 `$env:TRAIN_EPOCHS`)
- `--train-device cuda`: 학습 디바이스 (정책상 CPU 비허용)
- `--eval-device cuda`: 평가 디바이스 (정책상 CPU 비허용)

## 출력물

`reports/<dataset>/`에 통합 리포트를 생성합니다.
- `results.md` / `results.pdf`: 영문 리포트
- `results_KR.md` / `results_KR.pdf`: 한글 리포트
- `feature_importance_{dataset}.png`, `kinship_distribution_{dataset}.png`
- `<mode>/confusion_matrix_*.png`: 모드/모델별 혼동 행렬

모델 가중치는 `models/<dataset>/<mode>/{mlp.pth, cnn.pth}`에 저장됩니다.

## 참고 (클래스 불균형)
- 'zero' 모드는 재균형 미적용 기준선으로, 다수 클래스에 편향될 수 있습니다. 매크로/가중치 지표와 클래스별 결과를 함께 보세요.
- 'weighted'는 클래스 가중 손실, 'smote'는 학습 세트 과샘플링(검증은 원본 분포)입니다.
- AUC는 OvR 방식으로 강건하게 계산되며, 정의 불가 상황에서는 중립값(0.5)로 대체되어 N/A가 발생하지 않습니다.

## Git 트래킹
- `models/`, `reports/`, `data/processed/*.csv` 등 재생성 가능한/대용량 산출물은 git에서 무시됩니다.
- 필요 시 LFS를 설정해 산출물을 트래킹할 수 있습니다.

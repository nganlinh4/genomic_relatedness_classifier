### 프로젝트 계획: 친족 관계 분류

목표는 IBD 핵심 지표와 `merged_info.out`에서 파싱한 분포 통계를 결합해 친족 관계를 예측하는 것입니다. `cM_1`, `cM_3`, `cM_6` 세 데이터셋 각각에 대해 하나의 통합 리포트를 생성합니다(단일 리포트 내 교차-데이터셋 비교 없음).

---

#### 단계 1: 데이터 준비 (각 cM 데이터셋)

1. **`merged_info.out` 처리**
   - 각 라인을 `[pair] allChr key:value ...` 형태로 파싱하여 `pair`별 분포 통계 컬럼을 구성합니다.
   - 1개 `pair` = 1개 행의 DataFrame 생성 (전치(transpose) 불필요).

2. **`model_input_with_kinship_*.csv` 처리**
   - `model_input_with_kinship_filtered_<dataset>.csv` 로드.
   - 필수 컬럼만 유지: `pair`, `IBD1_len`, `IBD2_len`, `R1`, `R2`, `Num_Segs`, `Total_len`, `kinship`.

3. **병합**
   - 위 두 데이터를 `pair` 기준 내부 조인하여 `data/processed/merged_<dataset>.csv` 생성 (재생성 가능, git 무시).

#### 단계 2: EDA
- `kinship` 카테고리 분포 막대그래프 생성 → 클래스 불균형 확인.

#### 단계 3: 특성 선택 및 전처리
- 특성 X: IBD 6개 + 분포 통계 전체.
- 타깃 y: `kinship`.
- `StandardScaler`로 스케일링.
- RandomForest 중요도로 상위 50개 특성 선택; 선택 특성 전용 스케일러 저장.

#### 단계 4: 모델 학습 (CUDA 전용)
- 모드별 불균형 처리:
  - `zero`: 재균형 없음
  - `weighted`: 클래스 가중 손실
  - `smote`: 학습 세트에만 SMOTE 적용
- 모델 A: 고급 MLP (BatchNorm/Dropout 포함 심층 구조)
- 모델 B: 고급 1D-CNN (3개 Conv 블록 + 2개 Dense)

#### 단계 5: 평가 및 리포팅
- 지표: 정확도, F1(가중치/매크로), AUC(OvR: 가중치/매크로/마이크로; 강건 계산), 혼동 행렬
- RF 중요도 플롯 포함
- 불균형 유의사항: `zero`는 기준선; `weighted`/`smote` 검토 권장

#### 단계 6: 반복
- `cM_1`, `cM_3`, `cM_6`에 대해 1~5단계를 수행
- 산출물: `reports/<dataset>/results(.json/.md/.pdf)` 및 `results_KR(.md/.pdf)`

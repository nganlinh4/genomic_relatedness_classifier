flowchart TD
    subgraph "입력 데이터 수집"
        A1["model_input_with_kinship_filtered_{dataset}.csv<br/>게놈 쌍 관계<br/>친족 레이블: 1,2,3,4,5,6,UN"]
        A2["merged_info.out<br/>기본 통계적 특징<br/>103개 열"]
        A3["merged_added_info.out<br/>추가 통계적 특징<br/>거리 임계값 포함 123개 열"]
    end

    subgraph "데이터 통합 및 처리"
        B1[".out 파일 파싱<br/>강력한 키:값 추출"]
        B2["통계 데이터 병합<br/>열 충돌 처리"]
        B3["게놈 데이터와 통합<br/>최종 데이터셋: 124+개 열"]
    end

    subgraph "시나리오 생성"
        C1["포함 시나리오<br/>UN 레이블 포함 모든 레이블 유지<br/>~2,805개 샘플 (cM_1/cM_3)<br/>~2,787개 샘플 (cM_6)"]
        C2["NoUN 시나리오<br/>UN 레이블 제거<br/>모든 변형에서 ~882개 샘플<br/>클래스: 1,2,3,4,5,6만"]
    end

    subgraph "특징 엔지니어링 및 선택"
        D1["RandomForest 특징 선택<br/>상위 75개 특징"]
        D2["집계 특징 생성<br/>수학적 조합"]
        D3["특징 스케일링<br/>표준화 변환"]
    end

    subgraph "모델 훈련"
        E1["RandomForest<br/>전통적 앙상블"]
        E2["MLP<br/>5개 레이어 신경망"]
        E3["CNN<br/>1D 컨볼루션 네트워크"]

        F1["Zero 모드<br/>재균형 없음<br/>138.8-243.7초"]
        F2["Weighted 모드<br/>클래스 가중치 손실<br/>138.9-252.4초"]
        F3["SMOTE 모드<br/>합성 오버샘플링<br/>963.4-1,071.6초"]
        F4["OverUnder 모드<br/>SMOTE + ENN/Tomek<br/>197.1-244.7초"]
    end

    subgraph "모델 평가"
        G1["성능 메트릭<br/>F1, AUC, 정확도"]
        G2["혼동 행렬<br/>상세 분석"]
        G3["모델 비교<br/>변형당 24개 조합"]
    end

    subgraph "보고서 생성"
        H1["성능 순위<br/>RandomForest 일관되게 최고"]
        H2["시각화<br/>플롯 및 차트"]
        H3["다중 형식 출력<br/>마크다운, PDF, 이중 언어"]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> C1
    B3 --> C2

    C1 --> D1
    C2 --> D1
    D1 --> D2
    D2 --> D3

    D3 --> E1
    D3 --> E2
    D3 --> E3

    E1 --> F1
    E1 --> F2
    E1 --> F3
    E1 --> F4
    E2 --> F1
    E2 --> F2
    E2 --> F3
    E2 --> F4
    E3 --> F1
    E3 --> F2
    E3 --> F3
    E3 --> F4

    F1 --> G1
    F2 --> G1
    F3 --> G1
    F4 --> G1
    G1 --> G2
    G2 --> G3
    G3 --> H1
    H1 --> H2
    H2 --> H3

    style A1 fill:#e1f5fe
    style A2 fill:#e1f5fe
    style A3 fill:#e1f5fe
    style C1 fill:#fff3e0
    style C2 fill:#fff3e0
    style G1 fill:#e8f5e8
    style H1 fill:#f3e5f5
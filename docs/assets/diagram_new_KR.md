title 유전체 데이터 기반 친족 분류 모델 개발 플로우차트
direction right

// 입력 데이터 수집
Model Input with Kinship Filtered [shape: rectangle, color: lightblue, icon: file-text]
Merged Info Out [shape: rectangle, color: lightblue, icon: file-text]
Merged Added Info Out [shape: rectangle, color: lightblue, icon: file-text]

// 데이터 통합 및 처리
Parse Out Files [shape: rectangle, color: lightgrey, icon: code]
Merge Statistical Data [shape: rectangle, color: lightgrey, icon: columns]
Integrate with Genomic Data [shape: rectangle, color: lightgrey, icon: database]

// 시나리오 생성
Include Unrelated Scenario [shape: rectangle, color: orange, icon: layers]
No Unrelated Scenario [shape: rectangle, color: orange, icon: layers]

// 특징 엔지니어링 및 선택
Random Forest Feature Selection [shape: rectangle, color: yellow, icon: filter]
Aggregate Feature Creation [shape: rectangle, color: yellow, icon: plus-square]
Feature Scaling [shape: rectangle, color: yellow, icon: trending-up]

// 모델 훈련
Random Forest Model [shape: rectangle, color: green, icon: check-square]
Multi-Layer Perceptron Model [shape: rectangle, color: green, icon: cpu]
Convolutional Neural Network Model [shape: rectangle, color: green, icon: activity]

Zero Mode [shape: rectangle, color: green, icon: circle]
Weighted Mode [shape: rectangle, color: green, icon: circle]
SMOTE Mode [shape: rectangle, color: green, icon: circle]
OverUnder Mode [shape: rectangle, color: green, icon: circle]

// 모델 평가
Performance Metrics [shape: rectangle, color: lightgreen, icon: bar-chart-2]
Confusion Matrix [shape: rectangle, color: lightgreen, icon: grid]
Model Comparison [shape: rectangle, color: lightgreen, icon: list]

// 보고서 생성
Performance Ranking [shape: rectangle, color: violet, icon: award]
Visualization [shape: rectangle, color: violet, icon: pie-chart]
"Multi-format Output" [shape: rectangle, color: violet, icon: file]

// Relationships
Model Input with Kinship Filtered > Parse Out Files
Merged Info Out > Parse Out Files
Merged Added Info Out > Parse Out Files
Parse Out Files > Merge Statistical Data
Merge Statistical Data > Integrate with Genomic Data
Integrate with Genomic Data > Include Unrelated Scenario
Integrate with Genomic Data > No Unrelated Scenario

Include Unrelated Scenario > Random Forest Feature Selection
No Unrelated Scenario > Random Forest Feature Selection
Random Forest Feature Selection > Aggregate Feature Creation
Aggregate Feature Creation > Feature Scaling

Feature Scaling > Random Forest Model
Feature Scaling > Multi-Layer Perceptron Model
Feature Scaling > Convolutional Neural Network Model

Random Forest Model > Zero Mode
Random Forest Model > Weighted Mode
Random Forest Model > SMOTE Mode
Random Forest Model > OverUnder Mode
Multi-Layer Perceptron Model > Zero Mode
Multi-Layer Perceptron Model > Weighted Mode
Multi-Layer Perceptron Model > SMOTE Mode
Multi-Layer Perceptron Model > OverUnder Mode
Convolutional Neural Network Model > Zero Mode
Convolutional Neural Network Model > Weighted Mode
Convolutional Neural Network Model > SMOTE Mode
Convolutional Neural Network Model > OverUnder Mode

Zero Mode > Performance Metrics
Weighted Mode > Performance Metrics
SMOTE Mode > Performance Metrics
OverUnder Mode > Performance Metrics

Performance Metrics > Confusion Matrix
Confusion Matrix > Model Comparison
Model Comparison > Performance Ranking
Performance Ranking > Visualization
Visualization > "Multi-format Output"
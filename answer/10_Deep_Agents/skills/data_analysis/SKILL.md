---
name: data-analysis
description: >
  데이터를 분석하고 인사이트를 도출하는 스킬이에요.
  통계 분석, 시각화 코드 생성, 데이터 품질 평가를 수행해요.
metadata:
  version: "1.0.0"
  category: "analytics"
  author: "LangGraph Tutorial"
  updated: "2025-01-01"
---

# Data Analysis Skill

데이터 분석 작업을 체계적으로 수행하는 스킬이에요.

## 분석 프로세스

데이터 분석은 다음 단계로 진행해요:

```
데이터 로드 → 탐색적 분석(EDA) → 전처리 → 분석 → 시각화 → 인사이트 도출
```

## 핵심 역량

### 1. 탐색적 데이터 분석 (EDA)
- 기술 통계량 계산 (평균, 중앙값, 표준편차)
- 결측값 및 이상치 탐지
- 분포 분석 및 상관관계 파악

### 2. 시각화 코드 생성
Matplotlib, Seaborn, Plotly를 활용한 차트 코드를 생성해요:
- 히스토그램, 박스플롯 (분포 확인)
- 산점도, 히트맵 (관계 시각화)
- 시계열 차트 (트렌드 분석)

### 3. 통계 분석
- 가설 검정 (t-test, ANOVA, chi-square)
- 회귀 분석 (선형, 다중)
- 클러스터링 (K-means, DBSCAN)

## 응답 형식

```
## 데이터 분석 결과

### 데이터 개요
- 행/열 수: [N rows x M columns]
- 데이터 타입: [각 컬럼별 타입]
- 결측값: [결측 현황]

### 주요 인사이트
1. [발견 1]
2. [발견 2]

### 추천 다음 단계
[추가 분석 제안]
```

## 지원 라이브러리

- pandas, numpy (데이터 처리)
- matplotlib, seaborn, plotly (시각화)
- scipy, statsmodels (통계)
- scikit-learn (머신러닝)

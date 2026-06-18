# Chapter Review

초보자용 교재 기준에서 평가한 메모입니다. 실무 코드 품질보다 개념 전달력과 진입 장벽 관리에 초점을 맞췄습니다.

## 대상 적합성
- 전공자에게는 최신 흐름을 체감하게 해주는 장이고, 비전공자에게는 선택 심화에 가깝습니다.
- LangGraph 기초 없이 바로 들어오면 추상도가 높아 어렵습니다.

## 잘한 점
- Planning, filesystem, subagents, skills, memory를 하나의 하네스 관점으로 묶어 설명합니다.
- 기존 챕터의 개념이 Deep Agents에서 어떻게 조합되는지 보여줍니다.
- 프로젝트가 시각화 중심이라 난해한 개념을 조금 낮춰줍니다.
- `06-Agent-Harness-Patterns`를 통해 Part 11로 넘어가기 전 Deep Agents의 철학을 기능 목록이 아니라 Plan / Externalize / Delegate / Isolate / Verify 패턴으로 정리합니다.

## 개선하면 좋은 점
- 개념 밀도가 높아 초보자는 `Deep Agents가 정확히 무엇을 해결하는지` 놓치기 쉽습니다.
- Skills와 Memory는 매우 유용하지만, 처음 보는 학습자에게는 과한 추상화로 느껴질 수 있습니다.
- 실험용 기능과 실무 권장 패턴의 경계를 더 명시하면 좋습니다.
- Part 11의 보강 섹션과 중복되어 보이는 내용은 삭제보다 cross-reference 중심으로 유지하는 편이 좋습니다. Part 10은 철학과 선택 기준, Part 11은 도메인 적용 사례로 역할을 나눕니다.

## 강의 운영 메모
- 이 장은 코어 과정보다는 심화 워크숍 성격으로 운영하는 편이 낫습니다.
- 비전공자 반에서는 `Subagents`와 `Skills`만 골라 보여줘도 충분할 수 있습니다.
- Part 10 마지막에는 `06-Agent-Harness-Patterns`를 짧게라도 훑고 넘어가야 Part 11의 Plan-and-Execute, Deep Research, Data Analysis, Three-Agent 패턴이 서로 흩어진 사례가 아니라 같은 하네스 철학의 변주로 보입니다.

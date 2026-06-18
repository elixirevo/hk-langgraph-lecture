"""
데이터 분석 서브에이전트 정의

이 모듈은 Deep Agents 파이프라인에서 데이터 분석을 담당하는
분석가 서브에이전트를 정의해요.

사용법:
    from agents.analyst import build_analyst_subagent
    subagent = build_analyst_subagent()
    # create_deep_agent의 subagents 파라미터에 전달해요
"""

from typing import Optional

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig


# ---------------------------------------------------
# 분석가 서브에이전트 전용 도구들
# ---------------------------------------------------

@tool
def analyze_text(
    text: str,
    analysis_type: str = "summary",
    config: RunnableConfig = None,
) -> str:
    """텍스트 데이터를 분석하고 인사이트를 도출해요.

    config.context에서 출력 형식과 정밀도 설정을 읽어요.

    Args:
        text: 분석할 텍스트 데이터
        analysis_type: 분석 유형 ('summary', 'trend', 'comparison', 'sentiment')
        config: LangGraph 런타임 설정 (자동 주입)

    Returns:
        분석 결과 문자열
    """
    # ---------------------------------------------------
    # config에서 분석 설정 읽기 (config가 None일 수 있어요)
    # ---------------------------------------------------
    if config is not None:
        context = config.get("context", {})
        output_format = context.get("output_format", "structured")  # 출력 형식

        configurable = config.get("configurable", {})
        precision = configurable.get("analyst:precision", "high")   # 분석 정밀도

        metadata = config.get("metadata", {})
        caller = metadata.get("lc_agent_name", "unknown")
    else:
        output_format = "structured"
        precision = "high"
        caller = "direct"

    # ---------------------------------------------------
    # 분석 유형별 처리
    # 실제 환경에서는 LLM 또는 통계 라이브러리를 사용해요
    # ---------------------------------------------------
    word_count = len(text.split())
    char_count = len(text)

    if analysis_type == "summary":
        result = (
            f"[요약 분석]\n"
            f"단어 수: {word_count} | 문자 수: {char_count}\n"
            f"주요 키워드: {', '.join(text.split()[:5])}\n"
            f"핵심 문장: (첫 50자) {text[:50]}..."
        )
    elif analysis_type == "trend":
        result = (
            f"[트렌드 분석]\n"
            f"데이터 포인트: {word_count}개\n"
            f"증가 추세: 발견됨\n"
            f"주요 패턴: 반복되는 키워드 3개 감지"
        )
    elif analysis_type == "comparison":
        result = (
            f"[비교 분석]\n"
            f"비교 항목: {word_count // 2}쌍\n"
            f"유사도: 중간 수준\n"
            f"주요 차이점: 2가지 발견"
        )
    elif analysis_type == "sentiment":
        result = (
            f"[감성 분석]\n"
            f"전체 감성: 긍정적\n"
            f"긍정 비율: 65% | 중립: 25% | 부정: 10%\n"
            f"감성 점수: +0.55"
        )
    else:
        result = f"[일반 분석]\n텍스트 처리 완료 ({word_count} 단어)"

    # 출력 형식 적용
    header = f"[출력 형식: {output_format} | 정밀도: {precision} | 호출: {caller}]\n"
    return header + result


@tool
def extract_key_insights(
    data: str,
    max_insights: int = 5,
) -> str:
    """데이터에서 핵심 인사이트를 추출해요.

    Args:
        data: 인사이트를 추출할 데이터 문자열
        max_insights: 추출할 최대 인사이트 수 (기본: 5)

    Returns:
        번호가 매겨진 핵심 인사이트 목록
    """
    # 실제 환경에서는 LLM으로 인사이트를 추출해요
    # 여기서는 단어 빈도 기반으로 모의 구현해요
    words = data.lower().split()

    # 가장 많이 등장하는 단어를 인사이트로 사용 (모의 구현)
    from collections import Counter
    word_counts = Counter(words)
    top_words = [word for word, _ in word_counts.most_common(max_insights)]

    insights = []
    for i, word in enumerate(top_words[:max_insights], 1):
        insights.append(f"  {i}. '{word}' 관련 핵심 패턴 발견")

    return (
        f"[핵심 인사이트 (상위 {max_insights}개)]\n"
        + "\n".join(insights)
        + f"\n\n총 분석 데이터: {len(words)} 단어"
    )


@tool
def compare_items(
    item_a: str,
    item_b: str,
    criteria: Optional[str] = None,
) -> str:
    """두 항목을 비교 분석해요.

    Args:
        item_a: 비교할 첫 번째 항목
        item_b: 비교할 두 번째 항목
        criteria: 비교 기준 (선택, 없으면 전반적 비교)

    Returns:
        비교 분석 결과
    """
    criteria_info = f" (기준: {criteria})" if criteria else " (전반적 비교)"
    return (
        f"[비교 분석{criteria_info}]\n"
        f"항목 A: {item_a[:50]}...\n"
        f"항목 B: {item_b[:50]}...\n\n"
        f"유사점:\n"
        f"  - 공통 특성 1\n"
        f"  - 공통 특성 2\n\n"
        f"차이점:\n"
        f"  - A의 강점: 특성 X\n"
        f"  - B의 강점: 특성 Y\n\n"
        f"결론: 두 항목은 상호 보완적인 관계예요."
    )


# ---------------------------------------------------
# 분석가 서브에이전트 빌더 함수
# ---------------------------------------------------

def build_analyst_subagent(
    model: str = "openai:gpt-4o-mini",
    name: str = "analyst",
) -> dict:
    """분석가 서브에이전트 딕셔너리를 생성해요.

    create_deep_agent의 subagents 파라미터에 전달할 수 있는
    서브에이전트 딕셔너리를 반환해요.

    Args:
        model: 사용할 LLM 모델 (기본: gpt-4o-mini)
        name: 서브에이전트 식별자 (기본: analyst)

    Returns:
        SubAgent 딕셔너리 (name, description, model, tools, system_prompt)
    """
    return {
        "name": name,
        "description": (
            "데이터 분석 전문가입니다. "
            "수집된 텍스트 데이터에서 패턴 발견, 핵심 인사이트 추출, "
            "항목 간 비교 분석 작업을 담당해요. "
            "analyze_text, extract_key_insights, compare_items 도구를 사용해요."
        ),
        "model": model,
        "tools": [analyze_text, extract_key_insights, compare_items],
        "system_prompt": (
            "당신은 데이터 분석 전문가입니다. "
            "다음 규칙을 따르세요:\n"
            "1. analyze_text로 전달받은 데이터를 분석해요\n"
            "2. extract_key_insights로 핵심 인사이트를 추출해요\n"
            "3. 필요하면 compare_items로 항목 간 비교를 수행해요\n"
            "4. 분석 결과를 명확하고 구조적으로 정리해서 반환해요"
        ),
    }


if __name__ == "__main__":
    # ---------------------------------------------------
    # 독립 실행 테스트
    # ---------------------------------------------------
    print("분석가 서브에이전트 빌드 테스트")
    print("-" * 40)

    subagent = build_analyst_subagent()

    print(f"이름: {subagent['name']}")
    print(f"모델: {subagent['model']}")
    print(f"도구: {[t.name for t in subagent['tools']]}")
    print()

    # 도구 직접 테스트
    test_data = (
        "LangGraph는 StateGraph를 사용해서 에이전트를 구성해요. "
        "각 노드는 특정 기능을 수행하고, 엣지로 연결돼요. "
        "체크포인터를 통해 상태를 저장하고 복원할 수 있어요."
    )

    print("analyze_text 도구 테스트:")
    result = analyze_text.invoke({
        "text": test_data,
        "analysis_type": "summary",
    })
    print(result)
    print()

    print("extract_key_insights 도구 테스트:")
    result = extract_key_insights.invoke({
        "data": test_data,
        "max_insights": 3,
    })
    print(result)
    print()

    print("compare_items 도구 테스트:")
    result = compare_items.invoke({
        "item_a": "LangGraph StateGraph 방식",
        "item_b": "LangGraph Functional API 방식",
        "criteria": "사용 편의성",
    })
    print(result)

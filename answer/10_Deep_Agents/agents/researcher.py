"""
웹 리서처 서브에이전트 정의

이 모듈은 Deep Agents 파이프라인에서 정보 수집을 담당하는
리서처 서브에이전트를 정의해요.

사용법:
    from agents.researcher import build_researcher_subagent
    subagent = build_researcher_subagent()
    # create_deep_agent의 subagents 파라미터에 전달해요
"""

from typing import Optional

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig


# ---------------------------------------------------
# 리서처 서브에이전트 전용 도구들
# ---------------------------------------------------

@tool
def web_search(query: str, config: RunnableConfig) -> str:
    """웹에서 정보를 검색해요.

    config.context에서 언어 설정과 최대 검색 깊이를 읽어요.
    config.metadata의 lc_agent_name으로 호출 에이전트를 식별해요.

    Args:
        query: 검색할 질문 또는 키워드
        config: LangGraph 런타임 설정 (자동 주입)

    Returns:
        검색 결과 문자열
    """
    # ---------------------------------------------------
    # config에서 런타임 컨텍스트 읽기
    # ---------------------------------------------------
    context = config.get("context", {})
    language = context.get("language", "ko")       # 결과 언어 (기본: 한국어)
    max_results = context.get("max_results", 5)    # 최대 결과 수 (기본: 5)

    # 네임스페이스 키로 리서처 전용 설정 읽기
    configurable = config.get("configurable", {})
    max_depth = configurable.get("researcher:max_depth", 3)  # 검색 깊이

    # 호출 에이전트 이름 확인 (추적·디버깅용)
    metadata = config.get("metadata", {})
    caller = metadata.get("lc_agent_name", "unknown")

    # ---------------------------------------------------
    # 실제 검색 로직 (실습용 모의 구현)
    # 실제 환경에서는 TavilySearchResults, DuckDuckGoSearch 등을 사용해요
    # ---------------------------------------------------
    # from langchain_tavily import TavilySearch
    # tavily = TavilySearch(max_results=max_results)
    # results = tavily.invoke(query)

    # 모의 검색 결과 반환
    return (
        f"[웹 검색 결과]\n"
        f"검색어: {query}\n"
        f"언어: {language} | 최대 결과: {max_results} | 검색 깊이: {max_depth}\n"
        f"호출 에이전트: {caller}\n\n"
        f"결과 1: {query} 관련 주요 논문 발견 (2024)\n"
        f"결과 2: {query} 공식 문서 및 튜토리얼\n"
        f"결과 3: {query} 실전 예제 블로그 포스트\n"
        f"(실습용 모의 결과 — 실제 환경에서는 Tavily API 사용)"
    )


@tool
def fetch_webpage(url: str) -> str:
    """웹 페이지의 내용을 가져와요.

    Args:
        url: 내용을 읽을 웹 페이지 URL

    Returns:
        웹 페이지 텍스트 내용
    """
    # 실제 환경에서는 requests + BeautifulSoup 또는 Playwright 사용
    # import requests
    # from bs4 import BeautifulSoup
    # response = requests.get(url, timeout=10)
    # soup = BeautifulSoup(response.text, "html.parser")
    # return soup.get_text(separator="\n", strip=True)

    # 모의 구현
    return (
        f"[웹 페이지 내용]\n"
        f"URL: {url}\n"
        f"제목: 관련 기술 문서\n"
        f"내용: 이 페이지는 요청한 주제에 대한 상세 설명을 포함하고 있어요.\n"
        f"(실습용 모의 결과)"
    )


@tool
def summarize_sources(sources: str, focus_point: Optional[str] = None) -> str:
    """여러 출처의 내용을 요약해요.

    Args:
        sources: 요약할 출처 내용 (여러 결과를 합친 문자열)
        focus_point: 요약 시 집중할 포인트 (선택)

    Returns:
        핵심 내용 요약문
    """
    # 실제 환경에서는 LLM을 직접 호출해서 요약해요
    focus_info = f" (집중 포인트: {focus_point})" if focus_point else ""
    return (
        f"[출처 요약{focus_info}]\n"
        f"원본 길이: {len(sources)}자\n"
        f"핵심 내용:\n"
        f"  1. 첫 번째 핵심 포인트\n"
        f"  2. 두 번째 핵심 포인트\n"
        f"  3. 세 번째 핵심 포인트\n"
        f"(실습용 모의 요약)"
    )


# ---------------------------------------------------
# 리서처 서브에이전트 빌더 함수
# ---------------------------------------------------

def build_researcher_subagent(
    model: str = "openai:gpt-4o-mini",
    name: str = "researcher",
) -> dict:
    """리서처 서브에이전트 딕셔너리를 생성해요.

    create_deep_agent의 subagents 파라미터에 전달할 수 있는
    서브에이전트 딕셔너리를 반환해요.

    Args:
        model: 사용할 LLM 모델 (기본: gpt-4o-mini)
        name: 서브에이전트 식별자 (기본: researcher)

    Returns:
        SubAgent 딕셔너리 (name, description, model, tools, system_prompt)
    """
    return {
        "name": name,
        "description": (
            "웹 검색과 자료 수집 전문가입니다. "
            "특정 주제에 대한 정보 검색, 논문·블로그·공식 문서 수집, "
            "검색 결과 요약 작업을 담당해요. "
            "web_search, fetch_webpage, summarize_sources 도구를 사용해요."
        ),
        "model": model,
        "tools": [web_search, fetch_webpage, summarize_sources],
        "system_prompt": (
            "당신은 정보 수집 전문가입니다. "
            "다음 규칙을 따르세요:\n"
            "1. web_search로 관련 자료를 검색해요\n"
            "2. 중요한 URL은 fetch_webpage로 내용을 가져와요\n"
            "3. summarize_sources로 수집한 내용을 요약해요\n"
            "4. 결과를 구조적으로 정리해서 반환해요"
        ),
    }


if __name__ == "__main__":
    # ---------------------------------------------------
    # 독립 실행 테스트
    # ---------------------------------------------------
    print("리서처 서브에이전트 빌드 테스트")
    print("-" * 40)

    subagent = build_researcher_subagent()

    print(f"이름: {subagent['name']}")
    print(f"모델: {subagent['model']}")
    print(f"도구: {[t.name for t in subagent['tools']]}")
    print()

    # 도구 직접 테스트
    test_config: dict = {
        "context": {"language": "ko", "max_results": 3},
        "configurable": {"researcher:max_depth": 5},
        "metadata": {"lc_agent_name": "researcher"},
    }

    print("web_search 도구 테스트:")
    result = web_search.invoke({"query": "LangGraph V1 서브에이전트", "config": test_config})
    print(result)
    print()

    print("fetch_webpage 도구 테스트:")
    result = fetch_webpage.invoke({"url": "https://docs.langchain.com"})
    print(result)

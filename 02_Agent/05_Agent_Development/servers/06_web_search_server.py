"""
웹 검색 MCP 서버 (실습용)

이 모듈은 Tavily API를 사용하여 실시간 웹 검색 기능을
MCP 서버로 제공해요. 다른 MCP 서버들과 함께 사용하면
에이전트가 최신 정보를 검색할 수 있어요.

사용 방법:
    # stdio 방식으로 실행할 때
    uv run python servers/06_web_search_server.py

    # MCP 클라이언트 설정 예시
    server_configs = {
        "web_search": {
            "command": "uv",
            "args": ["run", "python", "servers/06_web_search_server.py"],
            "transport": "stdio",
        }
    }

환경변수:
    TAVILY_API_KEY: Tavily 검색 API 키 (필수)
"""

import os
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# 환경 변수 로드
load_dotenv()

# FastMCP 서버 초기화
# name: MCP 서버를 식별하는 이름이에요
# instructions: 클라이언트(LLM)에게 이 서버의 용도를 설명해요
mcp = FastMCP(
    "WebSearch",
    instructions=(
        "실시간 웹 검색 기능을 제공하는 서버예요. "
        "최신 뉴스, 정보, 기술 문서 등을 검색할 때 사용하세요."
    ),
)


@mcp.tool()
async def web_search(query: str, max_results: int = 3) -> str:
    """Tavily API를 사용하여 실시간 웹 검색을 수행해요.

    인터넷에서 최신 정보를 검색하고 결과를 요약하여 반환해요.
    뉴스, 기술 문서, 최신 이벤트 등 다양한 정보를 검색할 수 있어요.

    Args:
        query: 검색할 질의어 (한국어 또는 영어 가능)
        max_results: 반환할 최대 검색 결과 수 (기본값: 3, 최대: 5)

    Returns:
        검색 결과를 포맷팅한 문자열.
        각 결과에는 제목, URL, 내용 요약이 포함되어 있어요.

    Raises:
        ValueError: TAVILY_API_KEY가 설정되지 않은 경우
    """
    # Tavily API 키 확인
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return (
            "오류: TAVILY_API_KEY 환경변수가 설정되지 않았어요. "
            ".env 파일에 TAVILY_API_KEY=your_key_here 를 추가해주세요."
        )

    try:
        # langchain_tavily를 사용하여 검색 수행
        from langchain_tavily import TavilySearch

        # 검색 도구 초기화
        # max_results: 반환할 검색 결과 수를 제한해요
        search_tool = TavilySearch(max_results=min(max_results, 5))

        # 검색 실행
        results = await search_tool.ainvoke({"query": query})

        # 결과 포맷팅
        if not results:
            return f"'{query}'에 대한 검색 결과가 없어요."

        # 결과를 읽기 쉬운 형태로 정리해요
        formatted_results = [f"[웹 검색 결과: '{query}']\n"]

        if isinstance(results, list):
            # 리스트 형태의 결과 처리
            for i, result in enumerate(results[:max_results], 1):
                if isinstance(result, dict):
                    title = result.get("title", "제목 없음")
                    url = result.get("url", "")
                    content = result.get("content", "내용 없음")
                    # 너무 긴 내용은 자르기
                    if len(content) > 500:
                        content = content[:500] + "..."
                    formatted_results.append(
                        f"{i}. {title}\n   URL: {url}\n   {content}\n"
                    )
        elif isinstance(results, str):
            # 문자열 형태의 결과 처리
            formatted_results.append(results)

        return "\n".join(formatted_results)

    except ImportError:
        return (
            "오류: langchain_tavily 패키지가 설치되지 않았어요. "
            "'uv add langchain-tavily' 명령으로 설치해주세요."
        )
    except Exception as e:
        return f"검색 중 오류가 발생했어요: {str(e)}"


@mcp.tool()
async def search_news(topic: str, days_back: int = 7) -> str:
    """특정 주제에 대한 최신 뉴스를 검색해요.

    지정된 주제에 대해 최근 며칠 이내의 뉴스를 검색하고
    가장 관련성 높은 기사들을 요약하여 반환해요.

    Args:
        topic: 검색할 뉴스 주제 (예: "인공지능", "Python 3.12")
        days_back: 검색할 기간 (며칠 전까지, 기본값: 7일)

    Returns:
        최신 뉴스 결과를 포맷팅한 문자열.
        각 뉴스에는 제목, URL, 발행일, 요약이 포함되어 있어요.
    """
    # 뉴스 검색을 위한 특화된 질의어 생성
    # "site:news" 키워드로 뉴스 사이트에서 검색을 유도해요
    news_query = f"{topic} 최신 뉴스 {days_back}일"

    # API 키 확인
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return (
            "오류: TAVILY_API_KEY 환경변수가 설정되지 않았어요. "
            ".env 파일에 TAVILY_API_KEY=your_key_here 를 추가해주세요."
        )

    try:
        from langchain_tavily import TavilySearch

        # 뉴스 검색 도구 초기화 (결과 수 제한)
        search_tool = TavilySearch(max_results=5)

        # 뉴스 검색 실행
        results = await search_tool.ainvoke({"query": news_query})

        # 결과 포맷팅
        formatted_output = [f"['{topic}' 관련 최신 뉴스 (최근 {days_back}일)]\n"]

        if isinstance(results, list):
            for i, result in enumerate(results, 1):
                if isinstance(result, dict):
                    title = result.get("title", "제목 없음")
                    url = result.get("url", "")
                    content = result.get("content", "내용 없음")
                    published_date = result.get("published_date", "날짜 미상")

                    # 내용 길이 제한
                    if len(content) > 400:
                        content = content[:400] + "..."

                    formatted_output.append(
                        f"{i}. {title}\n"
                        f"   발행일: {published_date}\n"
                        f"   URL: {url}\n"
                        f"   {content}\n"
                    )
        elif isinstance(results, str):
            formatted_output.append(results)

        if len(formatted_output) == 1:
            return f"'{topic}'에 대한 최근 뉴스를 찾을 수 없어요."

        return "\n".join(formatted_output)

    except ImportError:
        return (
            "오류: langchain_tavily 패키지가 설치되지 않았어요. "
            "'uv add langchain-tavily' 명령으로 설치해주세요."
        )
    except Exception as e:
        return f"뉴스 검색 중 오류가 발생했어요: {str(e)}"


if __name__ == "__main__":
    # stdio 전송 방식으로 MCP 서버를 시작해요
    # stdio는 표준 입출력을 통해 클라이언트와 통신하므로
    # 별도의 포트 설정이 필요 없어요
    print("WebSearch MCP 서버가 시작되었어요 (stdio 방식)")
    print("제공 도구: web_search, search_news")
    mcp.run(transport="stdio")

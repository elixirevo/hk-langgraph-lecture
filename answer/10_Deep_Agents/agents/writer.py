"""
보고서 작성 서브에이전트 정의

이 모듈은 Deep Agents 파이프라인에서 보고서 작성을 담당하는
작성가 서브에이전트를 정의해요.

사용법:
    from agents.writer import build_writer_subagent
    subagent = build_writer_subagent()
    # create_deep_agent의 subagents 파라미터에 전달해요
"""

from typing import Literal, Optional

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig


# ---------------------------------------------------
# 작성가 서브에이전트 전용 도구들
# ---------------------------------------------------

@tool
def write_section(
    title: str,
    content: str,
    level: int = 2,
) -> str:
    """보고서의 섹션을 마크다운 형식으로 작성해요.

    Args:
        title: 섹션 제목
        content: 섹션 내용
        level: 제목 수준 (1=H1, 2=H2, 3=H3, 기본: 2)

    Returns:
        마크다운 형식의 섹션 문자열
    """
    # 제목 수준을 마크다운 헤딩으로 변환해요
    heading = "#" * max(1, min(level, 6))  # 1~6 사이로 제한
    return f"{heading} {title}\n\n{content}\n"


@tool
def format_report(
    sections: str,
    report_title: str = "분석 보고서",
    output_format: str = "markdown",
    config: RunnableConfig = None,
) -> str:
    """완성된 보고서를 지정한 형식으로 포맷해요.

    config.context에서 출력 언어와 형식 설정을 읽어요.

    Args:
        sections: 포맷할 보고서 내용 (여러 섹션을 합친 문자열)
        report_title: 보고서 제목 (기본: '분석 보고서')
        output_format: 출력 형식 ('markdown', 'html', 'plain', 기본: 'markdown')
        config: LangGraph 런타임 설정 (자동 주입)

    Returns:
        포맷된 보고서 문자열
    """
    # ---------------------------------------------------
    # config에서 출력 설정 읽기
    # ---------------------------------------------------
    if config is not None:
        context = config.get("context", {})
        language = context.get("language", "ko")             # 언어 설정
        ctx_format = context.get("output_format", output_format)  # 형식 (context 우선)
    else:
        language = "ko"
        ctx_format = output_format

    # ---------------------------------------------------
    # 형식별 보고서 생성
    # ---------------------------------------------------
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d")

    if ctx_format == "markdown":
        return (
            f"# {report_title}\n\n"
            f"> 언어: {language} | 생성일: {timestamp}\n\n"
            f"---\n\n"
            f"{sections}\n\n"
            f"---\n\n"
            f"*이 보고서는 Deep Agents 파이프라인에 의해 자동 생성되었어요.*"
        )
    elif ctx_format == "html":
        return (
            f"<html><body>\n"
            f"<h1>{report_title}</h1>\n"
            f"<p><em>언어: {language} | 생성일: {timestamp}</em></p>\n"
            f"<hr/>\n"
            f"{sections}\n"
            f"</body></html>"
        )
    else:  # plain
        return (
            f"=== {report_title} ===\n"
            f"언어: {language} | 생성일: {timestamp}\n"
            f"{'=' * 40}\n\n"
            f"{sections}"
        )


@tool
def add_table(
    headers: list[str],
    rows: list[list[str]],
    caption: Optional[str] = None,
) -> str:
    """마크다운 테이블을 생성해요.

    Args:
        headers: 테이블 헤더 목록 (예: ['항목', '설명', '예시'])
        rows: 데이터 행 목록 (각 행은 헤더 수만큼의 열을 가져야 해요)
        caption: 테이블 제목 (선택)

    Returns:
        마크다운 형식의 테이블 문자열
    """
    if not headers or not rows:
        return "(테이블 데이터가 없어요)"

    # 마크다운 테이블 헤더 생성
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"

    # 데이터 행 생성 (열 수가 헤더와 다르면 빈 값으로 채워요)
    data_rows = []
    for row in rows:
        # 헤더 수에 맞게 행 데이터 조정
        padded_row = list(row) + [""] * (len(headers) - len(row))
        padded_row = padded_row[:len(headers)]
        data_rows.append("| " + " | ".join(str(cell) for cell in padded_row) + " |")

    # 테이블 조합
    table_parts = []
    if caption:
        table_parts.append(f"**{caption}**\n")
    table_parts.extend([header_row, separator] + data_rows)

    return "\n".join(table_parts)


@tool
def create_executive_summary(
    full_content: str,
    max_points: int = 3,
) -> str:
    """전체 내용에서 핵심 요약(Executive Summary)을 생성해요.

    Args:
        full_content: 요약할 전체 보고서 내용
        max_points: 요약에 포함할 최대 핵심 포인트 수 (기본: 3)

    Returns:
        핵심 요약 섹션 문자열
    """
    # 실제 환경에서는 LLM으로 요약을 생성해요
    content_length = len(full_content)
    preview = full_content[:100].strip()

    # 핵심 포인트 생성 (모의 구현)
    points = []
    for i in range(1, max_points + 1):
        points.append(f"  {i}. 핵심 발견 {i}: 분석 결과의 주요 인사이트")

    summary_points = "\n".join(points)

    return (
        f"## 핵심 요약 (Executive Summary)\n\n"
        f"이 보고서는 {content_length}자 분량의 분석을 포함해요.\n\n"
        f"**주요 발견 사항:**\n"
        f"{summary_points}\n\n"
        f"**내용 미리보기:** {preview}...\n"
    )


# ---------------------------------------------------
# 작성가 서브에이전트 빌더 함수
# ---------------------------------------------------

def build_writer_subagent(
    model: str = "openai:gpt-4o-mini",
    name: str = "writer",
) -> dict:
    """작성가 서브에이전트 딕셔너리를 생성해요.

    create_deep_agent의 subagents 파라미터에 전달할 수 있는
    서브에이전트 딕셔너리를 반환해요.

    Args:
        model: 사용할 LLM 모델 (기본: gpt-4o-mini)
        name: 서브에이전트 식별자 (기본: writer)

    Returns:
        SubAgent 딕셔너리 (name, description, model, tools, system_prompt)
    """
    return {
        "name": name,
        "description": (
            "보고서 작성 전문가입니다. "
            "분석 결과를 바탕으로 구조적인 보고서 작성, "
            "마크다운 섹션 구성, 테이블 생성, 핵심 요약 작성 작업을 담당해요. "
            "write_section, format_report, add_table, create_executive_summary 도구를 사용해요."
        ),
        "model": model,
        "tools": [write_section, format_report, add_table, create_executive_summary],
        "system_prompt": (
            "당신은 보고서 작성 전문가입니다. "
            "다음 규칙을 따르세요:\n"
            "1. write_section으로 각 섹션을 마크다운 형식으로 작성해요\n"
            "2. 필요하면 add_table로 비교 테이블을 추가해요\n"
            "3. create_executive_summary로 핵심 요약을 생성해요\n"
            "4. format_report로 최종 보고서를 완성해요\n"
            "5. 명확하고 읽기 좋은 형태로 결과를 반환해요"
        ),
    }


# ---------------------------------------------------
# 전체 파이프라인 빌더: 세 서브에이전트를 한번에 구성
# ---------------------------------------------------

def build_pipeline_subagents(
    researcher_model: str = "openai:gpt-4o-mini",
    analyst_model: str = "openai:gpt-4o-mini",
    writer_model: str = "openai:gpt-4o-mini",
) -> list[dict]:
    """researcher → analyst → writer 파이프라인 서브에이전트 목록을 반환해요.

    create_deep_agent의 subagents 파라미터에 직접 전달할 수 있어요.

    Args:
        researcher_model: 리서처 에이전트 모델
        analyst_model: 분석가 에이전트 모델
        writer_model: 작성가 에이전트 모델

    Returns:
        [researcher_dict, analyst_dict, writer_dict] 리스트
    """
    # 임포트를 함수 내부에서 해서 순환 참조를 방지해요
    from agents.researcher import build_researcher_subagent
    from agents.analyst import build_analyst_subagent

    return [
        build_researcher_subagent(model=researcher_model),
        build_analyst_subagent(model=analyst_model),
        build_writer_subagent(model=writer_model),
    ]


if __name__ == "__main__":
    # ---------------------------------------------------
    # 독립 실행 테스트
    # ---------------------------------------------------
    print("작성가 서브에이전트 빌드 테스트")
    print("-" * 40)

    subagent = build_writer_subagent()

    print(f"이름: {subagent['name']}")
    print(f"모델: {subagent['model']}")
    print(f"도구: {[t.name for t in subagent['tools']]}")
    print()

    # 도구 직접 테스트

    print("write_section 도구 테스트:")
    section = write_section.invoke({
        "title": "LangGraph V1 분석",
        "content": "LangGraph V1은 StateGraph 기반의 에이전트 프레임워크예요.",
        "level": 2,
    })
    print(section)

    print("add_table 도구 테스트:")
    table = add_table.invoke({
        "headers": ["기능", "설명", "V0 여부"],
        "rows": [
            ["StateGraph", "그래프 기반 에이전트", "O"],
            ["Functional API", "함수형 API", "X (V1 신규)"],
            ["Middleware", "미들웨어 시스템", "X (V1 신규)"],
        ],
        "caption": "LangGraph V1 주요 기능",
    })
    print(table)
    print()

    print("format_report 도구 테스트:")
    report = format_report.invoke({
        "sections": section + "\n" + table,
        "report_title": "LangGraph 분석 보고서",
        "output_format": "markdown",
    })
    print(report[:300], "...")

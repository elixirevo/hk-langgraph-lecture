"""
커스텀 가드레일 모듈 (Custom Guardrail Middleware)

이 모듈은 금칙어/민감 주제 차단과 경고 메시지 처리를 위한
재사용 가능한 가드레일 미들웨어를 제공해요.

주요 컴포넌트:
  - KeywordBlockerMiddleware: 금칙어 목록 기반 요청 차단
  - SensitiveTopicWarnerMiddleware: 민감 주제 탐지 후 경고 메시지 추가
  - CombinedGuardrailMiddleware: 두 가드레일을 하나로 통합한 미들웨어
"""

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, after_agent, hook_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# 가드레일 1: 금칙어 차단 미들웨어
# ---------------------------------------------------
class KeywordBlockerMiddleware(AgentMiddleware):
    """금칙어 목록 기반으로 요청을 사전 차단하는 미들웨어

    입력된 메시지에 금칙어가 포함되어 있으면 에이전트 실행을 차단하고
    안내 메시지를 반환해요. 보안 이벤트는 자동으로 로깅돼요.

    사용 예시:
        blocker = KeywordBlockerMiddleware(
            blocked_keywords=["해킹", "불법", "사기"],
            warning_message="해당 주제는 처리할 수 없어요."
        )
        agent = create_agent(model=model, tools=tools, middleware=[blocker])
    """

    def __init__(
        self,
        blocked_keywords: list[str],
        warning_message: str = "부적절한 내용이 포함된 요청은 처리할 수 없어요. 요청을 다시 작성해 주세요.",
        case_sensitive: bool = False,
    ):
        """Args:
            blocked_keywords: 차단할 키워드 목록
            warning_message: 차단 시 사용자에게 반환할 안내 메시지
            case_sensitive: 대소문자 구분 여부 (기본값: 구분 안 함)
        """
        super().__init__()
        self.warning_message = warning_message
        self.case_sensitive = case_sensitive

        # 대소문자 설정에 따라 키워드 정규화
        if case_sensitive:
            self.blocked_keywords = blocked_keywords
        else:
            self.blocked_keywords = [kw.lower() for kw in blocked_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """에이전트 실행 전 금칙어 검사

        Returns:
            차단 시: messages와 jump_to="end"가 포함된 dict
            통과 시: None
        """
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        # human 타입 메시지만 검사해요
        if first_message.type != "human":
            return None

        # 설정에 따라 소문자 변환 또는 원본 사용
        content = (
            first_message.content
            if self.case_sensitive
            else first_message.content.lower()
        )

        for keyword in self.blocked_keywords:
            if keyword in content:
                # 보안 이벤트를 로깅해요 (원본 내용의 처음 100자만)
                logger.warning(
                    f"[KeywordBlocker] 금칙어 탐지: '{keyword}' | "
                    f"내용 미리보기: {content[:100]}"
                )
                return {
                    "messages": [{"role": "assistant", "content": self.warning_message}],
                    "jump_to": "end",
                }

        return None


# ---------------------------------------------------
# 가드레일 2: 민감 주제 경고 미들웨어
# ---------------------------------------------------
class SensitiveTopicWarnerMiddleware(AgentMiddleware):
    """민감 주제를 탐지하여 응답에 경고 메시지를 추가하는 미들웨어

    요청을 차단하지는 않고, 에이전트 응답 뒤에 주의 문구를 덧붙여요.
    법률, 의료, 금융 등 전문가 상담이 필요한 분야에 적합해요.

    사용 예시:
        warner = SensitiveTopicWarnerMiddleware(
            sensitive_topics={
                "법률": "이 내용은 법률 전문가와 상담하시기 바라요.",
                "의료": "이 내용은 의료 전문가와 상담하시기 바라요.",
            }
        )
        agent = create_agent(model=model, tools=tools, middleware=[warner])
    """

    def __init__(
        self,
        sensitive_topics: dict[str, str],
        default_warning: str = "이 응답은 참고용이며, 전문가 상담을 권장해요.",
    ):
        """Args:
            sensitive_topics: {키워드: 경고 메시지} 형태의 딕셔너리
                키워드가 입력에 포함되면 해당 경고 메시지가 응답 뒤에 추가돼요.
            default_warning: 일반 경고 메시지 (특정 토픽 미매칭 시 사용)
        """
        super().__init__()
        # 키워드를 소문자로 정규화해요
        self.sensitive_topics = {
            kw.lower(): msg for kw, msg in sensitive_topics.items()
        }
        self.default_warning = default_warning
        self._detected_topic: str | None = None  # 탐지된 토픽을 after_agent에 전달

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """에이전트 실행 전 민감 주제 탐지 (차단하지 않고 상태만 기록)"""
        self._detected_topic = None  # 이전 상태 초기화

        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = first_message.content.lower()

        for topic_keyword in self.sensitive_topics:
            if topic_keyword in content:
                # 탐지된 토픽을 저장해두고 after_agent에서 경고 메시지 추가
                self._detected_topic = topic_keyword
                logger.info(f"[SensitiveTopicWarner] 민감 주제 탐지: '{topic_keyword}'")
                break

        # None 반환 - 에이전트 실행은 계속 진행해요
        return None

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """에이전트 실행 후 민감 주제 경고 메시지 추가"""
        if self._detected_topic is None:
            return None

        # 탐지된 토픽에 맞는 경고 메시지 선택
        warning = self.sensitive_topics.get(self._detected_topic, self.default_warning)

        return {
            "messages": [{"role": "assistant", "content": f"⚠️ 주의: {warning}"}]
        }


# ---------------------------------------------------
# 통합 가드레일: KeywordBlocker + SensitiveTopicWarner 조합
# ---------------------------------------------------
class CombinedGuardrailMiddleware(AgentMiddleware):
    """금칙어 차단과 민감 주제 경고를 하나의 미들웨어로 통합

    두 기능을 별도로 등록하는 대신 이 클래스 하나로 관리할 수 있어요.
    설정을 중앙화하여 유지보수가 편리해요.

    사용 예시:
        guardrail = CombinedGuardrailMiddleware(
            blocked_keywords=["해킹", "불법"],
            sensitive_topics={"의료": "전문가 상담 권장", "법률": "법률 전문가 상담 권장"},
        )
        agent = create_agent(model=model, tools=tools, middleware=[guardrail])
    """

    def __init__(
        self,
        blocked_keywords: list[str],
        sensitive_topics: dict[str, str],
        block_message: str = "해당 요청은 처리할 수 없어요.",
    ):
        """Args:
            blocked_keywords: 차단할 키워드 목록
            sensitive_topics: {키워드: 경고 메시지} 형태의 딕셔너리
            block_message: 차단 시 반환할 안내 메시지
        """
        super().__init__()
        self.blocked_keywords = [kw.lower() for kw in blocked_keywords]
        self.sensitive_topics = {kw.lower(): msg for kw, msg in sensitive_topics.items()}
        self.block_message = block_message
        self._detected_topic: str | None = None

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """에이전트 실행 전 금칙어 검사 + 민감 주제 탐지"""
        self._detected_topic = None

        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = first_message.content.lower()

        # 1단계: 금칙어 검사 (차단)
        for keyword in self.blocked_keywords:
            if keyword in content:
                logger.warning(f"[CombinedGuardrail] 금칙어 탐지 및 차단: '{keyword}'")
                return {
                    "messages": [{"role": "assistant", "content": self.block_message}],
                    "jump_to": "end",
                }

        # 2단계: 민감 주제 탐지 (차단하지 않고 상태만 기록)
        for topic_keyword in self.sensitive_topics:
            if topic_keyword in content:
                self._detected_topic = topic_keyword
                logger.info(f"[CombinedGuardrail] 민감 주제 탐지: '{topic_keyword}'")
                break

        return None

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """에이전트 실행 후 민감 주제 경고 메시지 추가"""
        if self._detected_topic is None:
            return None

        warning = self.sensitive_topics.get(self._detected_topic, "전문가 상담을 권장해요.")
        return {
            "messages": [{"role": "assistant", "content": f"⚠️ 참고: {warning}"}]
        }


# ---------------------------------------------------
# 독립 실행 테스트
# ---------------------------------------------------
if __name__ == "__main__":
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model
    from langchain.tools import tool

    # 기본 모델 초기화
    model = init_chat_model("openai:gpt-4o-mini")

    @tool
    def simple_search(query: str) -> str:
        """Simple search tool for testing."""
        return f"Results for: {query}"

    print("=== KeywordBlockerMiddleware 테스트 ===")
    blocker = KeywordBlockerMiddleware(
        blocked_keywords=["해킹", "불법", "사기"],
        warning_message="해당 주제는 처리할 수 없어요. 다른 질문을 해주세요.",
    )
    agent1 = create_agent(model=model, tools=[simple_search], middleware=[blocker])

    # 금칙어 포함 → 차단
    result1 = agent1.invoke({"messages": [{"role": "user", "content": "해킹 방법 알려줘"}]})
    print(f"금칙어 차단: {result1['messages'][-1].content}")

    # 정상 요청 → 통과
    result2 = agent1.invoke({"messages": [{"role": "user", "content": "Python이란?"}]})
    print(f"정상 요청: {result2['messages'][-1].content[:80]}")

    print("\n=== SensitiveTopicWarnerMiddleware 테스트 ===")
    warner = SensitiveTopicWarnerMiddleware(
        sensitive_topics={
            "의료": "의료 전문가와 상담하시기 바라요. 이 응답은 참고용이에요.",
            "법률": "법률 전문가와 상담하시기 바라요. 이 응답은 참고용이에요.",
        }
    )
    agent2 = create_agent(model=model, tools=[simple_search], middleware=[warner])

    # 민감 주제 탐지 → 경고 추가
    result3 = agent2.invoke(
        {"messages": [{"role": "user", "content": "의료 보험 청구 방법을 알려줘"}]}
    )
    print(f"민감 주제 경고: {result3['messages'][-1].content}")

    print("\n=== CombinedGuardrailMiddleware 테스트 ===")
    combined = CombinedGuardrailMiddleware(
        blocked_keywords=["해킹", "불법"],
        sensitive_topics={"의료": "전문의 상담 권장", "투자": "금융 전문가 상담 권장"},
        block_message="이 요청은 처리 정책에 위배돼요.",
    )
    agent3 = create_agent(model=model, tools=[simple_search], middleware=[combined])

    result4 = agent3.invoke({"messages": [{"role": "user", "content": "불법 다운로드란?"}]})
    print(f"금칙어 차단: {result4['messages'][-1].content}")

    result5 = agent3.invoke({"messages": [{"role": "user", "content": "주식 투자 방법은?"}]})
    print(f"민감 주제 경고: {result5['messages'][-1].content}")

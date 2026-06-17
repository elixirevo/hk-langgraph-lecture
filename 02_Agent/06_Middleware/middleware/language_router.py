"""
언어 라우터 미들웨어 (Language Router Middleware)

이 모듈은 LangChain V1 에이전트의 입력 언어를 자동으로 감지하여
언어별 시스템 프롬프트로 전환하는 미들웨어를 구현해요.

주요 기능:
- 사용자 입력의 언어 자동 감지 (한국어, 영어, 일본어, 중국어 등)
- 언어별 사전 정의된 시스템 프롬프트 자동 적용
- 사용자 정의 언어-프롬프트 매핑 지원
- 감지 신뢰도 임계값 설정 (낮으면 기본 언어로 폴백)
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import SystemMessage


# 언어별 기본 시스템 프롬프트 템플릿
DEFAULT_PROMPTS: dict[str, str] = {
    "ko": (
        "당신은 친절하고 유능한 한국어 어시스턴트예요. "
        "항상 자연스럽고 정확한 한국어로 답변해요. "
        "답변은 명확하고 간결하게 작성해요."
    ),
    "en": (
        "You are a helpful and capable English assistant. "
        "Always respond in clear, natural English. "
        "Keep your answers concise and accurate."
    ),
    "ja": (
        "あなたは親切で有能な日本語アシスタントです。"
        "常に自然で正確な日本語で回答してください。"
        "回答は明確で簡潔に作成してください。"
    ),
    "zh": (
        "您是一位友善且有能力的中文助手。"
        "请始终用清晰、自然的中文回答。"
        "保持回答简洁准确。"
    ),
    "default": (
        "You are a helpful AI assistant. "
        "Respond in the same language as the user's input. "
        "Keep your answers clear and concise."
    ),
}


@dataclass
class LanguageDetectionResult:
    """언어 감지 결과를 담는 데이터 클래스예요.

    Attributes:
        language: 감지된 언어 코드 (예: 'ko', 'en', 'ja')
        confidence: 감지 신뢰도 (0.0 ~ 1.0)
        fallback: 신뢰도 임계값 미달로 기본 언어를 사용하면 True
    """

    language: str
    confidence: float
    fallback: bool = False


def detect_language(text: str, confidence_threshold: float = 0.6) -> LanguageDetectionResult:
    """텍스트의 언어를 감지해요.

    간단한 유니코드 범위 기반 감지 방법을 사용해요.
    프로덕션에서는 langdetect나 fasttext 등을 사용하는 것을 권장해요.

    Args:
        text: 감지할 텍스트
        confidence_threshold: 신뢰도 임계값. 이 값보다 낮으면 기본 언어로 폴백

    Returns:
        LanguageDetectionResult: 감지된 언어와 신뢰도
    """
    if not text or len(text.strip()) == 0:
        return LanguageDetectionResult(language="default", confidence=0.0, fallback=True)

    # 문자 수 카운터
    total_chars = len([c for c in text if not c.isspace()])
    if total_chars == 0:
        return LanguageDetectionResult(language="default", confidence=0.0, fallback=True)

    # 유니코드 범위로 언어 판별
    # 한국어: 한글 음절 (AC00-D7A3) + 한글 자모 (1100-11FF, 3130-318F)
    korean_chars = len(re.findall(r"[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]", text))

    # 일본어: 히라가나 (3040-309F) + 카타카나 (30A0-30FF)
    japanese_chars = len(re.findall(r"[\u3040-\u309F\u30A0-\u30FF]", text))

    # 중국어: CJK 통합 한자 (4E00-9FFF)
    # 참고: 한자는 중국어, 일본어, 한국어에 공통 사용되므로
    # 다른 언어 문자가 없을 때만 중국어로 판단해요
    chinese_chars = len(re.findall(r"[\u4E00-\u9FFF]", text))

    # 영어: ASCII 알파벳
    english_chars = len(re.findall(r"[a-zA-Z]", text))

    # 언어별 비율 계산
    ratios = {
        "ko": korean_chars / total_chars,
        "ja": japanese_chars / total_chars,
        "en": english_chars / total_chars,
    }

    # 한자만 있고 다른 언어 특수문자가 없으면 중국어로 판단
    if chinese_chars > 0 and korean_chars == 0 and japanese_chars == 0:
        ratios["zh"] = chinese_chars / total_chars

    # 가장 높은 비율의 언어 선택
    best_lang = max(ratios, key=lambda k: ratios[k])
    best_confidence = ratios[best_lang]

    # 신뢰도가 임계값보다 낮으면 기본 언어로 폴백
    if best_confidence < confidence_threshold:
        return LanguageDetectionResult(
            language="default",
            confidence=best_confidence,
            fallback=True,
        )

    return LanguageDetectionResult(language=best_lang, confidence=best_confidence)


@dataclass
class LanguageRouterConfig:
    """언어 라우터 설정을 담는 데이터 클래스예요.

    Attributes:
        custom_prompts: 언어 코드 → 시스템 프롬프트 매핑. DEFAULT_PROMPTS를 덮어써요
        confidence_threshold: 언어 감지 신뢰도 임계값 (기본: 0.6)
        verbose: True이면 감지된 언어를 출력해요 (기본: False)
        fallback_language: 감지 실패 시 사용할 기본 언어 (기본: 'default')
    """

    custom_prompts: dict[str, str] = field(default_factory=dict)
    confidence_threshold: float = 0.6
    verbose: bool = False
    fallback_language: str = "default"


class LanguageRouterMiddleware(AgentMiddleware):
    """입력 언어를 자동 감지하여 언어별 시스템 프롬프트를 적용하는 미들웨어예요.

    AgentMiddleware를 상속하여 매 모델 호출 전에 사용자의 최근 메시지를
    분석하고 적절한 시스템 프롬프트를 주입해요.

    Args:
        custom_prompts: 기본 프롬프트를 덮어쓸 언어별 커스텀 프롬프트
        confidence_threshold: 언어 감지 신뢰도 임계값 (0.0 ~ 1.0)
        verbose: True이면 감지 결과를 콘솔에 출력
        fallback_language: 감지 실패 시 사용할 언어 코드

    Examples:
        >>> router = LanguageRouterMiddleware(verbose=True)
        >>> agent = create_agent(model=model, tools=[...], middleware=[router])

        >>> # 커스텀 프롬프트 설정
        >>> router = LanguageRouterMiddleware(
        ...     custom_prompts={
        ...         "ko": "한국 고객을 위한 전문적인 금융 상담 어시스턴트예요.",
        ...         "en": "You are a professional financial advisor for international clients.",
        ...     }
        ... )
    """

    def __init__(
        self,
        custom_prompts: dict[str, str] | None = None,
        confidence_threshold: float = 0.6,
        verbose: bool = False,
        fallback_language: str = "default",
    ):
        self.config = LanguageRouterConfig(
            custom_prompts=custom_prompts or {},
            confidence_threshold=confidence_threshold,
            verbose=verbose,
            fallback_language=fallback_language,
        )

    def _detect_language_from_request(self, request: ModelRequest) -> str:
        """요청 메시지에서 마지막 사용자 메시지의 언어를 감지해요."""
        last_user_text = ""
        for msg in reversed(request.messages):
            msg_type = getattr(msg, "type", None)
            msg_role = msg.get("role") if isinstance(msg, dict) else None

            if msg_type == "human" or msg_role == "user":
                last_user_text = (
                    msg["content"]
                    if isinstance(msg, dict)
                    else str(getattr(msg, "content", ""))
                )
                break

        detection = detect_language(last_user_text, self.config.confidence_threshold)

        if self.config.verbose:
            lang_display = detection.language
            fallback_note = " (폴백)" if detection.fallback else ""
            print(
                f"[LanguageRouter] 감지 언어: {lang_display}{fallback_note} "
                f"(신뢰도: {detection.confidence:.2f})"
            )

        # 사용할 언어 결정 (폴백 처리)
        effective_lang = detection.language
        if detection.fallback:
            effective_lang = self.config.fallback_language

        # 커스텀 프롬프트 우선, 없으면 기본 프롬프트 사용
        prompt = (
            self.config.custom_prompts.get(effective_lang)
            or DEFAULT_PROMPTS.get(effective_lang)
            or DEFAULT_PROMPTS["default"]
        )
        return prompt

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable,
    ) -> ModelResponse:
        """모델 호출 전에 언어별 시스템 프롬프트를 주입해요."""
        prompt = self._detect_language_from_request(request)
        # system_prompt 필드를 통해 시스템 프롬프트 주입
        updated_request = request.override(system_message=SystemMessage(content=prompt))
        return handler(updated_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable,
    ) -> ModelResponse:
        """비동기: 모델 호출 전에 언어별 시스템 프롬프트를 주입해요."""
        prompt = self._detect_language_from_request(request)
        updated_request = request.override(system_message=SystemMessage(content=prompt))
        return await handler(updated_request)

    def get_supported_languages(self) -> list[str]:
        """지원하는 언어 코드 목록을 반환해요.

        Returns:
            지원하는 언어 코드 리스트 (예: ['ko', 'en', 'ja', 'zh', 'default'])
        """
        all_langs = set(DEFAULT_PROMPTS.keys())
        all_langs.update(self.config.custom_prompts.keys())
        return sorted(all_langs)

    def add_language(self, language_code: str, prompt: str) -> None:
        """새로운 언어와 시스템 프롬프트를 동적으로 추가해요.

        Args:
            language_code: 언어 코드 (예: 'fr', 'de', 'es')
            prompt: 해당 언어의 시스템 프롬프트
        """
        self.config.custom_prompts[language_code] = prompt


if __name__ == "__main__":
    # 독립 실행 테스트
    from dotenv import load_dotenv

    load_dotenv()

    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model
    from langchain.tools import tool

    # 테스트용 도구
    @tool
    def get_info(topic: str) -> str:
        """주제에 대한 정보를 제공해요."""
        return f"{topic}에 대한 정보: 이것은 간단한 설명이에요."

    model = init_chat_model("openai:gpt-4o-mini")

    # 기본 언어 라우터 설정 (verbose 모드)
    router = LanguageRouterMiddleware(verbose=True)

    print(f"지원 언어: {router.get_supported_languages()}")
    print()

    agent = create_agent(
        model=model,
        tools=[get_info],
        middleware=[router],
    )

    # 한국어 테스트
    print("=== 한국어 입력 ===")
    result_ko = agent.invoke(
        {"messages": [{"role": "user", "content": "LangGraph가 뭔가요?"}]}
    )
    print("응답:", result_ko["messages"][-1].content[:150])
    print()

    # 영어 테스트
    print("=== English Input ===")
    result_en = agent.invoke(
        {"messages": [{"role": "user", "content": "What is LangGraph?"}]}
    )
    print("Response:", result_en["messages"][-1].content[:150])
    print()

    # 커스텀 프롬프트 추가 후 테스트
    print("=== 커스텀 프롬프트 추가 ===")
    router.add_language(
        "ko",
        "당신은 친절한 금융 전문 어시스턴트예요. 항상 투자 위험을 명시해요.",
    )
    result_custom = agent.invoke(
        {"messages": [{"role": "user", "content": "주식 투자 전략을 알려줘요"}]}
    )
    print("커스텀 응답:", result_custom["messages"][-1].content[:150])

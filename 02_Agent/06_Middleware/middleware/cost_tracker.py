"""
비용 추적 미들웨어 (Cost Tracker Middleware)

이 모듈은 LangChain V1 에이전트의 토큰 사용량과 API 호출 비용을
실시간으로 추적하는 미들웨어를 구현해요.

주요 기능:
- 모델 호출별 입력/출력 토큰 수 추적
- gpt-4o-mini / gpt-4o 가격 기준 비용 계산
- 예산(budget_usd) 초과 시 에이전트 자동 중단
- 실행 종료 후 비용 보고서 출력
"""

from typing import Any, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime


# 모델별 가격 테이블 (1K 토큰당 USD 기준, 2024년 기준)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {
        "input_per_1k": 0.000150,   # $0.000150 / 1K 입력 토큰
        "output_per_1k": 0.000600,  # $0.000600 / 1K 출력 토큰
    },
    "gpt-4o": {
        "input_per_1k": 0.002500,   # $0.002500 / 1K 입력 토큰
        "output_per_1k": 0.010000,  # $0.010000 / 1K 출력 토큰
    },
    "claude-sonnet-4-5": {
        "input_per_1k": 0.003000,
        "output_per_1k": 0.015000,
    },
    "claude-haiku-4-5": {
        "input_per_1k": 0.000800,
        "output_per_1k": 0.004000,
    },
    # 기본 가격 (알 수 없는 모델에 적용)
    "default": {
        "input_per_1k": 0.001000,
        "output_per_1k": 0.003000,
    },
}


class CostTrackerMiddleware(AgentMiddleware):
    """토큰 사용량과 API 호출 비용을 실시간으로 추적하는 미들웨어예요.

    에이전트가 실행되는 동안 각 모델 호출의 토큰 수와 예상 비용을
    누적하고, 예산 초과 시 에이전트를 자동으로 중단해요.

    Args:
        model_name: 가격 계산에 사용할 모델 이름 (기본: gpt-4o-mini)
        budget_usd: 최대 허용 비용 (달러). None이면 무제한 (기본: None)
        verbose: True이면 각 호출마다 비용을 실시간 출력 (기본: True)

    Examples:
        >>> tracker = CostTrackerMiddleware(budget_usd=0.10)
        >>> agent = create_agent(model=model, tools=[...], middleware=[tracker])
        >>> agent.invoke({"messages": [...]})
        >>> tracker.print_cost_report()
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        budget_usd: float | None = None,
        verbose: bool = True,
    ):
        super().__init__()
        # 가격 정보 로드 (알 수 없는 모델이면 default 가격 사용)
        self.pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["default"])
        self.model_name = model_name
        self.budget_usd = budget_usd
        self.verbose = verbose

        # 누적 통계
        self.call_count = 0              # 총 모델 호출 횟수
        self.total_input_tokens = 0      # 총 입력 토큰 수
        self.total_output_tokens = 0     # 총 출력 토큰 수
        self.total_cost_usd = 0.0        # 총 누적 비용 (USD)
        self.call_history: list[dict[str, Any]] = []  # 호출별 상세 기록

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """모델 호출 전: 예산 초과 여부를 확인해요."""
        # 예산이 설정되어 있고 이미 초과한 경우
        if self.budget_usd is not None and self.total_cost_usd >= self.budget_usd:
            if self.verbose:
                print(
                    f"\n[CostTracker] 예산 초과! "
                    f"${self.total_cost_usd:.4f} >= ${self.budget_usd:.4f}"
                )
            # 예산 초과 안내 메시지를 반환하여 에이전트를 중단해요
            from langchain.messages import AIMessage

            return {
                "messages": [
                    AIMessage(
                        f"예산 한도(${self.budget_usd:.2f})에 도달했어요. "
                        f"지금까지의 비용: ${self.total_cost_usd:.4f}"
                    )
                ]
            }
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """모델 호출을 감싸서 토큰 사용량과 비용을 추적해요."""
        self.call_count += 1

        # 실제 모델 호출
        response = handler(request)

        # 응답에서 토큰 사용량 추출 (usage_metadata가 있는 경우)
        input_tokens = 0
        output_tokens = 0

        # 마지막 메시지에서 usage_metadata 추출 시도
        if hasattr(response, "messages") and response.messages:
            last_msg = response.messages[-1]
            if hasattr(last_msg, "usage_metadata") and last_msg.usage_metadata:
                usage = last_msg.usage_metadata
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

        # 토큰 수를 추정할 수 없는 경우: 메시지 길이로 대략적으로 추정
        # (4자 ≈ 1 토큰 기준)
        if input_tokens == 0:
            total_chars = sum(
                len(str(m.content)) for m in request.messages if hasattr(m, "content")
            )
            input_tokens = max(1, total_chars // 4)

        if output_tokens == 0 and hasattr(response, "messages") and response.messages:
            last_msg = response.messages[-1]
            output_tokens = max(1, len(str(getattr(last_msg, "content", ""))) // 4)

        # 비용 계산 (1K 토큰 단위)
        call_cost = (
            (input_tokens / 1000) * self.pricing["input_per_1k"]
            + (output_tokens / 1000) * self.pricing["output_per_1k"]
        )

        # 누적 통계 업데이트
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += call_cost

        # 호출 기록 저장
        self.call_history.append(
            {
                "call_number": self.call_count,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": call_cost,
                "cumulative_cost_usd": self.total_cost_usd,
            }
        )

        # 실시간 비용 출력
        if self.verbose:
            budget_info = ""
            if self.budget_usd is not None:
                remaining = self.budget_usd - self.total_cost_usd
                budget_info = f" | 잔여 예산: ${remaining:.4f}"
            print(
                f"[CostTracker] 호출 #{self.call_count} "
                f"| 입력: {input_tokens}t, 출력: {output_tokens}t "
                f"| 이번 비용: ${call_cost:.5f} "
                f"| 누적: ${self.total_cost_usd:.5f}"
                f"{budget_info}"
            )

        return response

    def get_stats(self) -> dict[str, Any]:
        """누적 통계를 딕셔너리로 반환해요.

        Returns:
            통계 딕셔너리:
            - total_calls: 총 모델 호출 횟수
            - total_input_tokens: 총 입력 토큰 수
            - total_output_tokens: 총 출력 토큰 수
            - total_tokens: 총 토큰 수 (입력 + 출력)
            - total_cost_usd: 총 비용 (달러)
            - budget_usd: 설정된 예산 (None이면 무제한)
            - budget_remaining: 잔여 예산 (None이면 무제한)
            - call_history: 각 호출의 상세 기록 목록
        """
        budget_remaining = None
        if self.budget_usd is not None:
            budget_remaining = max(0.0, self.budget_usd - self.total_cost_usd)

        return {
            "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "budget_usd": self.budget_usd,
            "budget_remaining": budget_remaining,
            "call_history": self.call_history,
        }

    def print_cost_report(self) -> None:
        """비용 보고서를 보기 좋게 출력해요."""
        stats = self.get_stats()

        print()
        print("=" * 55)
        print("비용 추적 보고서 (Cost Tracker Report)")
        print("=" * 55)
        print(f"모델:             {self.model_name}")
        print(f"총 호출 횟수:     {stats['total_calls']}회")
        print(f"입력 토큰:        {stats['total_input_tokens']:,}개")
        print(f"출력 토큰:        {stats['total_output_tokens']:,}개")
        print(f"총 토큰:          {stats['total_tokens']:,}개")
        print("-" * 55)
        print(f"총 비용:          ${stats['total_cost_usd']:.5f}")

        if stats["budget_usd"] is not None:
            usage_pct = (stats["total_cost_usd"] / stats["budget_usd"]) * 100
            print(f"설정 예산:        ${stats['budget_usd']:.5f}")
            print(f"예산 사용률:      {usage_pct:.1f}%")
            print(f"잔여 예산:        ${stats['budget_remaining']:.5f}")

        if stats["call_history"]:
            print("-" * 55)
            print("호출별 비용 내역:")
            for record in stats["call_history"]:
                print(
                    f"  #{record['call_number']:02d}: "
                    f"입력 {record['input_tokens']:>5}t, "
                    f"출력 {record['output_tokens']:>5}t, "
                    f"비용 ${record['cost_usd']:.5f}"
                )

        print("=" * 55)


if __name__ == "__main__":
    # 독립 실행 테스트
    from dotenv import load_dotenv

    load_dotenv()

    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model
    from langchain.tools import tool

    # 테스트용 도구
    @tool
    def search(query: str) -> str:
        """간단한 검색 도구예요."""
        return f"'{query}' 검색 결과: 관련 정보 발견"

    # 비용 추적기 생성 ($0.05 예산)
    tracker = CostTrackerMiddleware(
        model_name="gpt-4o-mini",
        budget_usd=0.05,
        verbose=True,
    )

    # 에이전트 생성
    model = init_chat_model("openai:gpt-4o-mini")
    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[tracker],
    )

    print("=== 비용 추적 테스트 ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "AI 에이전트란 무엇인가요?"}]}
    )

    print("\n응답:", result["messages"][-1].content[:100])

    # 비용 보고서 출력
    tracker.print_cost_report()

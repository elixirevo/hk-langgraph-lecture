"""
로깅 미들웨어 (Logging Middleware)

이 모듈은 LangChain V1 에이전트의 모든 모델/도구 호출을 로깅하는
미들웨어를 구현해요.

주요 기능:
- 모델 호출 전/후 로깅 (before_model, after_model)
- 도구 호출 감싸기 (wrap_tool_call)
- 호출 통계 수집 (횟수, 소요 시간, 성공/실패)
- 구조화된 로그 출력 (JSON 형태 지원)
"""

import time
import json
from datetime import datetime
from typing import Any, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime


class LoggingMiddleware(AgentMiddleware):
    """모든 모델과 도구 호출을 로깅하는 미들웨어예요.

    에이전트 실행 중 발생하는 모든 모델 호출과 도구 실행을
    기록하고 통계를 수집해요.

    Args:
        verbose: True이면 상세 내용 출력, False이면 요약만 출력
        log_to_file: 파일 경로를 지정하면 해당 파일에도 로그 저장
        max_content_length: 출력할 최대 내용 길이 (기본: 100자)
    """

    def __init__(
        self,
        verbose: bool = True,
        log_to_file: str | None = None,
        max_content_length: int = 100,
    ):
        super().__init__()
        self.verbose = verbose
        self.log_to_file = log_to_file
        self.max_content_length = max_content_length

        # 통계 수집용 내부 변수
        self.model_call_count = 0      # 총 모델 호출 횟수
        self.tool_call_count = 0       # 총 도구 호출 횟수
        self.total_model_time = 0.0    # 총 모델 응답 시간 (초)
        self.total_tool_time = 0.0     # 총 도구 실행 시간 (초)
        self.errors = []               # 발생한 오류 목록
        self._model_start_time = None  # 모델 호출 시작 시간 (내부 사용)

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """모델 호출 전에 실행돼요. 시작 시간을 기록하고 로그를 출력해요."""
        self._model_start_time = time.time()  # 시작 시간 기록
        self.model_call_count += 1

        msg_count = len(state["messages"])
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # 밀리초 포함

        log_entry = {
            "event": "model_call_start",
            "timestamp": timestamp,
            "call_number": self.model_call_count,
            "message_count": msg_count,
        }

        if self.verbose:
            print(f"\n{'='*50}")
            print(f"[{timestamp}] 모델 호출 #{self.model_call_count} 시작")
            print(f"  현재 메시지 수: {msg_count}개")

            # 마지막 사용자 메시지 미리보기
            last_human_msg = self._get_last_human_message(state["messages"])
            if last_human_msg:
                preview = last_human_msg[:self.max_content_length]
                print(f"  마지막 사용자 입력: {preview}...")
        else:
            print(f"[모델 #{self.model_call_count}] 시작 (메시지 {msg_count}개)")

        self._write_log(log_entry)
        return None  # 상태 변경 없음

    def after_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """모델 호출 후에 실행돼요. 소요 시간과 응답 내용을 로깅해요."""
        elapsed = 0.0
        if self._model_start_time is not None:
            elapsed = time.time() - self._model_start_time
            self.total_model_time += elapsed

        last_msg = state["messages"][-1]
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # 응답 타입 확인 (AI 메시지, 도구 호출 요청 등)
        msg_type = type(last_msg).__name__
        content_str = str(last_msg.content)
        preview = content_str[:self.max_content_length]

        # 도구 호출 여부 확인
        has_tool_calls = (
            hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0
        )

        log_entry = {
            "event": "model_call_end",
            "timestamp": timestamp,
            "call_number": self.model_call_count,
            "elapsed_seconds": round(elapsed, 3),
            "message_type": msg_type,
            "has_tool_calls": has_tool_calls,
        }

        if self.verbose:
            print(f"[{timestamp}] 모델 호출 #{self.model_call_count} 완료")
            print(f"  소요 시간: {elapsed:.3f}초")
            print(f"  응답 타입: {msg_type}")
            if has_tool_calls:
                tool_names = [tc["name"] for tc in last_msg.tool_calls]
                print(f"  도구 호출 요청: {', '.join(tool_names)}")
            else:
                print(f"  응답 미리보기: {preview}...")
        else:
            print(
                f"[모델 #{self.model_call_count}] 완료 "
                f"({elapsed:.2f}초, 도구호출={has_tool_calls})"
            )

        self._write_log(log_entry)
        return None

    def wrap_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        handler: Callable[[], Any],
    ) -> Any:
        """도구 호출을 감싸서 실행 전후에 로깅해요."""
        self.tool_call_count += 1
        start_time = time.time()
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        log_start = {
            "event": "tool_call_start",
            "timestamp": timestamp,
            "tool_name": tool_name,
            "call_number": self.tool_call_count,
            "args": tool_args,
        }

        if self.verbose:
            print(f"\n[{timestamp}] 도구 호출 #{self.tool_call_count}: {tool_name}")
            print(f"  인자: {json.dumps(tool_args, ensure_ascii=False)}")
        else:
            print(f"[도구 #{self.tool_call_count}] {tool_name} 실행 중...")

        self._write_log(log_start)

        # 도구 실행 (오류 처리 포함)
        try:
            result = handler()  # 실제 도구 호출
            elapsed = time.time() - start_time
            self.total_tool_time += elapsed

            timestamp_end = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            result_preview = str(result)[:self.max_content_length]

            log_end = {
                "event": "tool_call_end",
                "timestamp": timestamp_end,
                "tool_name": tool_name,
                "call_number": self.tool_call_count,
                "elapsed_seconds": round(elapsed, 3),
                "success": True,
                "result_preview": result_preview,
            }

            if self.verbose:
                print(f"[{timestamp_end}] 도구 완료: {tool_name} ({elapsed:.3f}초)")
                print(f"  결과: {result_preview}...")
            else:
                print(f"[도구 #{self.tool_call_count}] {tool_name} 완료 ({elapsed:.2f}초)")

            self._write_log(log_end)
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)
            self.errors.append(
                {"tool": tool_name, "error": error_msg, "timestamp": timestamp}
            )

            log_error = {
                "event": "tool_call_error",
                "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "tool_name": tool_name,
                "call_number": self.tool_call_count,
                "elapsed_seconds": round(elapsed, 3),
                "success": False,
                "error": error_msg,
            }

            print(f"[오류] 도구 {tool_name} 실패: {error_msg}")
            self._write_log(log_error)
            raise  # 오류 재전파

    def get_stats(self) -> dict[str, Any]:
        """수집된 통계를 반환해요.

        Returns:
            통계 딕셔너리:
            - model_calls: 총 모델 호출 횟수
            - tool_calls: 총 도구 호출 횟수
            - avg_model_time: 평균 모델 응답 시간 (초)
            - avg_tool_time: 평균 도구 실행 시간 (초)
            - error_count: 발생한 오류 수
        """
        avg_model = (
            self.total_model_time / self.model_call_count
            if self.model_call_count > 0
            else 0
        )
        avg_tool = (
            self.total_tool_time / self.tool_call_count
            if self.tool_call_count > 0
            else 0
        )

        return {
            "model_calls": self.model_call_count,
            "tool_calls": self.tool_call_count,
            "total_model_time": round(self.total_model_time, 3),
            "total_tool_time": round(self.total_tool_time, 3),
            "avg_model_time": round(avg_model, 3),
            "avg_tool_time": round(avg_tool, 3),
            "error_count": len(self.errors),
            "errors": self.errors,
        }

    def print_summary(self) -> None:
        """실행 통계 요약을 출력해요."""
        stats = self.get_stats()
        print("\n" + "=" * 50)
        print("로깅 미들웨어 실행 통계")
        print("=" * 50)
        print(f"모델 호출 횟수: {stats['model_calls']}회")
        print(f"도구 호출 횟수: {stats['tool_calls']}회")
        print(f"총 모델 시간:   {stats['total_model_time']:.3f}초")
        print(f"총 도구 시간:   {stats['total_tool_time']:.3f}초")
        print(f"평균 모델 응답: {stats['avg_model_time']:.3f}초")
        print(f"평균 도구 실행: {stats['avg_tool_time']:.3f}초")
        print(f"오류 발생 수:   {stats['error_count']}건")
        if stats["errors"]:
            print("\n오류 목록:")
            for err in stats["errors"]:
                print(f"  - {err['tool']}: {err['error']}")
        print("=" * 50)

    def _get_last_human_message(self, messages: list) -> str | None:
        """메시지 목록에서 마지막 사용자 메시지 내용을 반환해요."""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                return str(msg.content)
            # dict 형태의 메시지 처리
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
        return None

    def _write_log(self, log_entry: dict[str, Any]) -> None:
        """로그 항목을 파일에 기록해요 (log_to_file이 설정된 경우)."""
        if self.log_to_file:
            try:
                with open(self.log_to_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[경고] 로그 파일 쓰기 실패: {e}")


if __name__ == "__main__":
    # 독립 실행 테스트
    from langchain.chat_models import init_chat_model
    from langchain.tools import tool
    from langchain.agents import create_agent

    # 테스트용 도구 정의
    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Sunny in {city}!"

    # 로거 인스턴스 생성
    logger = LoggingMiddleware(verbose=True, max_content_length=80)

    # 에이전트 생성
    model = init_chat_model("openai:gpt-4o-mini")
    agent = create_agent(
        model=model,
        tools=[get_weather],
        middleware=[logger],
    )

    # 테스트 실행
    print("=== 로깅 미들웨어 테스트 ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "서울 날씨 알려줘"}]}
    )

    print("\n최종 응답:", result["messages"][-1].content)

    # 통계 출력
    logger.print_summary()

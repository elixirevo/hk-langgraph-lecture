"""
현재 시간을 제공하는 MCP 서버 (streamable-http 전송 방식)

이 서버는 HTTP 엔드포인트를 통해 시간 조회 도구를 제공해요.
독립 프로세스로 실행되며, 클라이언트가 HTTP로 연결해요.

실행 방법:
    uv run python servers/02_time_server.py
    # 또는
    python servers/02_time_server.py

클라이언트 연결 URL: http://127.0.0.1:8002/mcp
"""

from mcp.server.fastmcp import FastMCP
from typing import Optional
import pytz
from datetime import datetime

# FastMCP 서버 인스턴스를 생성해요
# port: HTTP 서버가 사용할 포트 번호예요
mcp = FastMCP(
    "CurrentTime",
    instructions="시간대를 지정하면 현재 시간을 알려주는 서버예요.",
    port=8002,  # HTTP 전송 시 사용할 포트
)


@mcp.tool()
async def get_current_time(timezone: Optional[str] = "Asia/Seoul") -> str:
    """지정한 시간대의 현재 시간을 조회해요.

    Args:
        timezone: 조회할 시간대 (기본값: Asia/Seoul)
                  예: America/New_York, Europe/London, Asia/Tokyo

    Returns:
        해당 시간대의 현재 시간 문자열 (YYYY-MM-DD HH:MM:SS TZ 형식)
    """
    try:
        # pytz로 시간대 객체를 가져와요
        tz = pytz.timezone(timezone)

        # 해당 시간대의 현재 시간을 계산해요
        current_time = datetime.now(tz)

        # 읽기 좋은 형식으로 변환해요
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")

        return f"현재 {timezone} 시간: {formatted_time}"

    except pytz.exceptions.UnknownTimeZoneError:
        return f"오류: '{timezone}'은 올바른 시간대가 아니에요. 예: Asia/Seoul, America/New_York"
    except Exception as e:
        return f"시간 조회 중 오류 발생: {str(e)}"


if __name__ == "__main__":
    # 시간 MCP 서버 시작 중... (포트 8002)
    # 클라이언트 연결 URL: http://127.0.0.1:8002/mcp
    # 종료: Ctrl+C

    # streamable-http 전송 방식으로 서버를 시작해요
    # 이 서버는 독립 프로세스로 계속 실행되어야 해요
    mcp.run(transport="streamable-http")
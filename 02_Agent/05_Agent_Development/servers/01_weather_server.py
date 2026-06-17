"""
날씨 정보를 제공하는 MCP 서버 (stdio 전송 방식)

이 서버는 FastMCP를 사용해서 날씨 조회 도구를 제공해요.
stdio 방식으로 실행되어 클라이언트가 subprocess로 관리해요.
"""

from mcp.server.fastmcp import FastMCP

# FastMCP 서버 인스턴스를 생성해요
# instructions는 LLM이 이 서버의 목적을 이해하는 데 도움을 줘요
mcp = FastMCP(
    "Weather",
    instructions="날씨 정보를 제공하는 서버예요. 도시 이름을 주면 현재 날씨를 알려줘요.",
)


@mcp.tool()
async def get_weather(location: str) -> str:
    """지정한 도시의 현재 날씨를 조회해요.

    Args:
        location: 날씨를 조회할 도시 이름 (예: 서울, 부산, Tokyo)

    Returns:
        해당 도시의 현재 날씨 정보 문자열
    """
    # 실제 서비스에서는 날씨 API(OpenWeatherMap 등)를 호출해야 해요
    # 여기서는 교육용으로 시뮬레이션 응답을 반환해요
    return f"It's always Sunny in {location}"


if __name__ == "__main__":
    # stdio 전송 방식으로 서버를 시작해요
    # 클라이언트가 이 프로세스의 stdin/stdout을 통해 통신해요
    mcp.run(transport="stdio")
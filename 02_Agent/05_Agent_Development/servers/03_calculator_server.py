"""
사칙연산을 제공하는 MCP 서버 (stdio 전송 방식)

이 서버는 4가지 수학 연산 도구를 제공해요:
- add: 덧셈
- subtract: 뺄셈
- multiply: 곱셈
- divide: 나눗셈
"""

from mcp.server.fastmcp import FastMCP

# 계산기 서버 인스턴스를 생성해요
mcp = FastMCP(
    "Calculator",
    instructions="수학 연산을 수행하는 계산기 서버예요. 덧셈, 뺄셈, 곱셈, 나눗셈을 지원해요.",
)


@mcp.tool()
async def add(a: float, b: float) -> str:
    """두 숫자를 더해요.

    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자

    Returns:
        덧셈 결과 문자열
    """
    result = a + b
    return f"{a} + {b} = {result}"


@mcp.tool()
async def subtract(a: float, b: float) -> str:
    """첫 번째 숫자에서 두 번째 숫자를 빼요.

    Args:
        a: 첫 번째 숫자 (피감수)
        b: 두 번째 숫자 (감수)

    Returns:
        뺄셈 결과 문자열
    """
    result = a - b
    return f"{a} - {b} = {result}"


@mcp.tool()
async def multiply(a: float, b: float) -> str:
    """두 숫자를 곱해요.

    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자

    Returns:
        곱셈 결과 문자열
    """
    result = a * b
    return f"{a} × {b} = {result}"


@mcp.tool()
async def divide(a: float, b: float) -> str:
    """첫 번째 숫자를 두 번째 숫자로 나눠요.

    Args:
        a: 나뉨수 (분자)
        b: 나누는 수 (분모, 0이 되면 오류 반환)

    Returns:
        나눗셈 결과 문자열. b가 0이면 오류 메시지를 반환해요.
    """
    # 0으로 나누기를 방지해요
    if b == 0:
        return "오류: 0으로 나눌 수 없어요!"
    result = a / b
    return f"{a} ÷ {b} = {result:.4f}"


if __name__ == "__main__":
    # stdio 전송 방식으로 서버를 시작해요
    mcp.run(transport="stdio")
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MemoServer", instructions="메모를 저장하고 조회하는 서버예요.")
_memos = {}

@mcp.tool()
async def save_memo(title: str, content: str) -> str:
    """새 메모를 저장해요."""
    _memos[title] = content
    return f"메모 '{title}'이(가) 저장되었어요!"

@mcp.tool()
async def get_memo(title: str) -> str:
    """저장된 메모를 조회해요."""
    if title not in _memos:
        return f"메모 '{title}'이(가) 없어요."
    return f"[{title}] {_memos[title]}"

@mcp.tool()
async def list_memos() -> str:
    """저장된 모든 메모 제목을 조회해요."""
    if not _memos:
        return "저장된 메모가 없어요."
    return f"저장된 메모 목록: {list(_memos.keys())}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
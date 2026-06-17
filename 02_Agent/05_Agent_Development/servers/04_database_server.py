"""
데이터베이스 MCP 서버 (Database MCP Server)

이 모듈은 SQLite 기반 샘플 데이터베이스에 접근하는 MCP 서버예요.
FastMCP를 사용해 쿼리 실행, 테이블 조회, 데이터 삽입 도구를 제공해요.

주요 기능:
- SQL 쿼리 실행 (SELECT, INSERT, UPDATE, DELETE)
- 테이블 목록 조회
- 스키마 정보 확인

교육 목적: MCP 서버가 데이터베이스 접근을 추상화하는 방법을 보여줘요.
"""

import sqlite3
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------
# FastMCP 서버 초기화
# ---------------------------------------------------
# 서버 이름과 설명으로 MCP 서버를 초기화해요
mcp = FastMCP(
    "Database",
    instructions="SQLite 데이터베이스에 접근하여 데이터를 조회하고 관리하는 어시스턴트예요.",
)

# ---------------------------------------------------
# SQLite 데이터베이스 초기화
# ---------------------------------------------------
# 실제 프로덕션에서는 파일 경로나 연결 문자열을 사용해요
DB_PATH = Path(__file__).with_name("sample_database.sqlite3")
ALLOWED_COLUMNS = {
    "employees": {"name", "department", "salary", "hire_date"},
    "products": {"name", "category", "price", "stock"},
}
ALLOWED_TABLES = set(ALLOWED_COLUMNS)


def get_connection() -> sqlite3.Connection:
    """SQLite 데이터베이스 연결을 반환하는 헬퍼 함수예요.

    Returns:
        sqlite3.Connection: 데이터베이스 연결 객체
    """
    conn = sqlite3.connect(DB_PATH)
    # 딕셔너리 형태로 결과를 반환하도록 설정해요
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    """샘플 데이터로 데이터베이스를 초기화해요.

    테이블 생성과 초기 데이터 삽입을 수행해요.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # 직원 테이블 생성
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary INTEGER NOT NULL,
            hire_date TEXT NOT NULL
        )
    """)

    # 제품 테이블 생성
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER NOT NULL DEFAULT 0
        )
    """)

    # 샘플 직원 데이터 삽입
    cursor.execute("DELETE FROM employees")
    employees = [
        (1, "김철수", "개발팀", 5000000, "2022-01-15"),
        (2, "이영희", "마케팅팀", 4500000, "2021-06-01"),
        (3, "박민준", "개발팀", 5500000, "2020-03-10"),
        (4, "정수진", "인사팀", 4200000, "2023-02-28"),
        (5, "최동훈", "영업팀", 4800000, "2021-11-15"),
    ]
    cursor.executemany(
        "INSERT INTO employees (id, name, department, salary, hire_date) VALUES (?, ?, ?, ?, ?)",
        employees,
    )

    # 샘플 제품 데이터 삽입
    cursor.execute("DELETE FROM products")
    products = [
        (1, "노트북 Pro", "전자제품", 1500000, 50),
        (2, "무선 마우스", "전자제품", 45000, 200),
        (3, "기계식 키보드", "전자제품", 180000, 80),
        (4, "모니터 27인치", "전자제품", 350000, 30),
        (5, "USB 허브", "주변기기", 35000, 150),
    ]
    cursor.executemany(
        "INSERT INTO products (id, name, category, price, stock) VALUES (?, ?, ?, ?, ?)",
        products,
    )

    conn.commit()
    conn.close()


# 서버 시작 시 데이터베이스를 초기화해요
initialize_database()


# ---------------------------------------------------
# MCP 도구 정의
# ---------------------------------------------------


@mcp.tool()
async def execute_query(sql: str) -> str:
    """SQL 쿼리를 실행하고 결과를 반환해요.

    학생용 서버에서는 안전을 위해 SELECT 조회만 허용해요.
    데이터 추가는 insert_record 도구를 사용하세요.

    Args:
        sql: 실행할 SQL 쿼리 문자열

    Returns:
        쿼리 실행 결과 문자열
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        normalized = sql.strip().rstrip(";")
        if ";" in normalized or not normalized.upper().startswith("SELECT "):
            conn.close()
            return "안전한 SELECT 단일 쿼리만 실행할 수 있어요."

        cursor.execute(normalized)

        rows = cursor.fetchall()
        if not rows:
            conn.close()
            return "결과가 없습니다."

        # 컬럼 이름 가져오기
        columns = [description[0] for description in cursor.description]

        # 테이블 형식으로 결과를 포맷팅해요
        result_lines = [" | ".join(columns)]
        result_lines.append("-" * len(result_lines[0]))

        for row in rows:
            result_lines.append(" | ".join(str(value) for value in row))

        conn.close()
        return "\n".join(result_lines)

    except sqlite3.Error as e:
        # SQLite 오류 발생 시 오류 메시지를 반환해요
        return f"데이터베이스 오류: {str(e)}"


@mcp.tool()
async def list_tables() -> str:
    """데이터베이스에 있는 모든 테이블 목록을 반환해요.

    Returns:
        테이블 이름 목록 문자열
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # SQLite 시스템 테이블에서 사용자 테이블 목록을 가져와요
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        conn.close()

        if not tables:
            return "테이블이 없습니다."

        table_names = [table[0] for table in tables]
        return f"테이블 목록 ({len(table_names)}개):\n" + "\n".join(
            f"  - {name}" for name in table_names
        )

    except sqlite3.Error as e:
        return f"데이터베이스 오류: {str(e)}"


@mcp.tool()
async def get_table_schema(table_name: str) -> str:
    """특정 테이블의 스키마 정보를 반환해요.

    Args:
        table_name: 스키마를 조회할 테이블 이름

    Returns:
        테이블 스키마 정보 문자열 (컬럼명, 타입, 제약조건)
    """
    try:
        if table_name not in ALLOWED_TABLES:
            return f"테이블 '{table_name}'은 조회할 수 없습니다."

        conn = get_connection()
        cursor = conn.cursor()

        # 테이블 존재 여부 확인
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
        )
        if not cursor.fetchone():
            conn.close()
            return f"테이블 '{table_name}'이 존재하지 않습니다."

        # PRAGMA로 컬럼 정보를 가져와요
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        conn.close()

        # 스키마 정보를 포맷팅해요
        schema_lines = [f"테이블: {table_name}", "-" * 40]
        for col in columns:
            # col = (cid, name, type, notnull, dflt_value, pk)
            col_id, col_name, col_type, not_null, default_val, is_pk = col
            constraints = []
            if is_pk:
                constraints.append("PRIMARY KEY")
            if not_null:
                constraints.append("NOT NULL")
            if default_val is not None:
                constraints.append(f"DEFAULT {default_val}")

            constraint_str = ", ".join(constraints) if constraints else ""
            schema_lines.append(f"  {col_name}: {col_type} {constraint_str}".strip())

        return "\n".join(schema_lines)

    except sqlite3.Error as e:
        return f"데이터베이스 오류: {str(e)}"


@mcp.tool()
async def insert_record(table_name: str, data: dict[str, Any]) -> str:
    """테이블에 새 레코드를 삽입해요.

    Args:
        table_name: 데이터를 삽입할 테이블 이름
        data: 삽입할 데이터 딕셔너리 (컬럼명: 값)

    Returns:
        삽입 결과 메시지 (생성된 레코드 ID 포함)
    """
    try:
        if table_name not in ALLOWED_TABLES:
            return f"테이블 '{table_name}'에는 삽입할 수 없습니다."

        allowed = ALLOWED_COLUMNS[table_name]
        invalid_columns = sorted(set(data) - allowed)
        if invalid_columns:
            return "허용되지 않은 컬럼: " + ", ".join(invalid_columns)

        conn = get_connection()
        cursor = conn.cursor()

        # 컬럼과 값을 분리해요
        columns = list(data.keys())
        values = list(data.values())
        placeholders = ", ".join(["?" for _ in columns])
        column_str = ", ".join(columns)

        sql = f"INSERT INTO {table_name} ({column_str}) VALUES ({placeholders})"
        cursor.execute(sql, values)

        # 새로 생성된 레코드의 ID를 가져와요
        new_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return f"레코드 삽입 완료: ID={new_id}로 '{table_name}'에 저장되었습니다."

    except sqlite3.Error as e:
        return f"데이터베이스 오류: {str(e)}"


if __name__ == "__main__":
    print("데이터베이스 MCP 서버 시작 중...")
    print(f"사용 가능한 도구: execute_query, list_tables, get_table_schema, insert_record")

    # stdio 전송 방식으로 MCP 서버를 시작해요
    mcp.run(transport="stdio")

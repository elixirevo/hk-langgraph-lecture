"""
파일 처리 도구 모듈

CSV, JSON, TXT 파일을 읽고 변환하는 도구 함수들을 제공해요.
LangChain @tool 데코레이터와 함께 사용할 수 있어요.

지원 형식:
    - CSV: 헤더 포함 tabular 데이터
    - JSON: 단일 객체 또는 배열
    - TXT: 일반 텍스트
"""

import csv
import json
from pathlib import Path
from typing import Literal, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

WORKSPACE_ROOT = (Path(__file__).resolve().parent / "workspace").resolve()
WORKSPACE_ROOT.mkdir(exist_ok=True)


def resolve_workspace_path(file_path: str) -> Path:
    """학생용 workspace 디렉터리 안의 경로만 허용해요."""
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate
    resolved = candidate.resolve()
    if resolved != WORKSPACE_ROOT and WORKSPACE_ROOT not in resolved.parents:
        raise ValueError("파일 경로는 tools/workspace 디렉터리 안에 있어야 해요.")
    return resolved


class FileReadInput(BaseModel):
    """파일 읽기 도구의 입력 스키마예요."""

    file_path: str = Field(
        description="tools/workspace 안의 상대 경로"
    )
    encoding: str = Field(
        default="utf-8",
        description="파일 인코딩 (기본값: utf-8, 한국어 파일은 euc-kr도 시도해요)"
    )
    max_rows: int = Field(
        default=50,
        description="CSV 파일에서 반환할 최대 행 수 (기본값: 50)",
        ge=1,
        le=1000
    )


class FileConvertInput(BaseModel):
    """파일 변환 도구의 입력 스키마예요."""

    source_path: str = Field(description="변환할 원본 파일 경로")
    target_path: str = Field(description="저장할 대상 파일 경로")
    target_format: Literal["json", "csv", "txt"] = Field(
        description="변환 목표 형식: 'json', 'csv', 'txt'"
    )


def read_csv(file_path: Path, encoding: str, max_rows: int) -> str:
    """CSV 파일을 읽고 테이블 형태의 문자열로 반환해요.

    Args:
        file_path: CSV 파일 경로
        encoding: 파일 인코딩
        max_rows: 반환할 최대 행 수

    Returns:
        테이블 형태 문자열
    """
    with open(file_path, encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)

    if not rows:
        return "CSV 파일이 비어 있어요"

    headers = list(rows[0].keys())
    header_line = " | ".join(headers)
    separator = "-" * len(header_line)

    data_lines = []
    for row in rows:
        line = " | ".join(str(row.get(h, "")) for h in headers)
        data_lines.append(line)

    total_count = f"\n\n총 {len(rows)}행 반환됨 (최대 {max_rows}행 제한)"
    return f"{header_line}\n{separator}\n" + "\n".join(data_lines) + total_count


def read_json(file_path: Path, encoding: str) -> str:
    """JSON 파일을 읽고 들여쓰기된 문자열로 반환해요.

    Args:
        file_path: JSON 파일 경로
        encoding: 파일 인코딩

    Returns:
        들여쓰기 적용된 JSON 문자열
    """
    with open(file_path, encoding=encoding) as f:
        data = json.load(f)

    # 배열인 경우 처음 10개만 반환
    if isinstance(data, list) and len(data) > 10:
        preview = data[:10]
        result = json.dumps(preview, ensure_ascii=False, indent=2)
        return result + f"\n\n... (총 {len(data)}개 중 10개 표시)"

    return json.dumps(data, ensure_ascii=False, indent=2)


def read_txt(file_path: Path, encoding: str, max_chars: int = 5000) -> str:
    """TXT 파일을 읽고 텍스트를 반환해요.

    Args:
        file_path: TXT 파일 경로
        encoding: 파일 인코딩
        max_chars: 반환할 최대 문자 수

    Returns:
        파일 텍스트 내용
    """
    with open(file_path, encoding=encoding) as f:
        content = f.read()

    if len(content) > max_chars:
        return content[:max_chars] + f"\n\n... (전체 {len(content)}자 중 {max_chars}자 표시)"
    return content


@tool(args_schema=FileReadInput)
def read_file(
    file_path: str,
    encoding: str = "utf-8",
    max_rows: int = 50
) -> str:
    """로컬 파일을 읽어서 내용을 반환해요.

    CSV, JSON, TXT 파일을 지원해요. CSV는 테이블 형태로,
    JSON은 들여쓰기된 형태로, TXT는 그대로 반환해요.

    Args:
        file_path: 읽을 파일 경로 (CSV, JSON, TXT 지원)
        encoding: 파일 인코딩 (기본: utf-8)
        max_rows: CSV 최대 행 수 (기본: 50)

    Returns:
        파일 내용을 형식에 맞게 변환한 문자열
    """
    try:
        path = resolve_workspace_path(file_path)
    except ValueError as e:
        return str(e)

    if not path.exists():
        return f"파일을 찾을 수 없어요: '{file_path}'"

    if not path.is_file():
        return f"'{file_path}'는 파일이 아니에요 (디렉토리일 수 있어요)"

    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            return read_csv(path, encoding, max_rows)
        elif suffix == ".json":
            return read_json(path, encoding)
        elif suffix in (".txt", ".md", ".log"):
            return read_txt(path, encoding)
        else:
            # 알 수 없는 형식은 TXT로 시도
            return read_txt(path, encoding)
    except UnicodeDecodeError:
        return f"인코딩 오류: '{encoding}'으로 읽을 수 없어요. 다른 인코딩을 시도해보세요 (예: euc-kr)"
    except Exception as e:
        return f"파일 읽기 오류: {e}"


@tool
def get_file_info(file_path: str) -> str:
    """파일의 기본 정보(크기, 형식, 수정일)를 반환해요.

    Args:
        file_path: 정보를 조회할 파일 경로

    Returns:
        파일 이름, 크기, 확장자, 최종 수정일 정보
    """
    import datetime

    try:
        path = resolve_workspace_path(file_path)
    except ValueError as e:
        return str(e)

    if not path.exists():
        return f"파일을 찾을 수 없어요: '{file_path}'"

    stat = path.stat()
    size_kb = stat.st_size / 1024
    modified = datetime.datetime.fromtimestamp(stat.st_mtime)

    return (
        f"파일명: {path.name}\n"
        f"크기: {size_kb:.1f} KB ({stat.st_size:,} bytes)\n"
        f"확장자: {path.suffix or '없음'}\n"
        f"최종 수정: {modified.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"workspace 기준 경로: {path.relative_to(WORKSPACE_ROOT)}"
    )


@tool(args_schema=FileConvertInput)
def convert_file(
    source_path: str,
    target_path: str,
    target_format: str
) -> str:
    """파일을 다른 형식으로 변환해요 (CSV ↔ JSON 지원).

    Args:
        source_path: 원본 파일 경로 (CSV 또는 JSON)
        target_path: 저장할 파일 경로
        target_format: 변환 목표 형식 ('json', 'csv', 'txt')

    Returns:
        변환 성공/실패 메시지
    """
    try:
        source = resolve_workspace_path(source_path)
        target = resolve_workspace_path(target_path)
    except ValueError as e:
        return str(e)

    if not source.exists():
        return f"원본 파일을 찾을 수 없어요: '{source_path}'"

    try:
        target.parent.mkdir(parents=True, exist_ok=True)

        # CSV → JSON 변환
        if source.suffix.lower() == ".csv" and target_format == "json":
            with open(source, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                data = list(reader)

            with open(target, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return f"CSV → JSON 변환 완료: {target.relative_to(WORKSPACE_ROOT)} ({len(data)}행)"

        # JSON → CSV 변환
        elif source.suffix.lower() == ".json" and target_format == "csv":
            with open(source, encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list) or not data:
                return "JSON 파일이 배열 형태가 아니거나 비어 있어요"

            with open(target, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

            return f"JSON → CSV 변환 완료: {target.relative_to(WORKSPACE_ROOT)} ({len(data)}행)"

        else:
            return f"'{source.suffix}' → '{target_format}' 변환은 아직 지원하지 않아요"

    except Exception as e:
        return f"변환 오류: {e}"


if __name__ == "__main__":
    # 독립 실행 테스트: 임시 CSV 파일 생성 후 읽기
    import os

    print("=== 파일 프로세서 테스트 ===")

    # 테스트용 CSV 파일 생성
    temp_csv_path = WORKSPACE_ROOT / "sample.csv"
    with open(temp_csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["이름", "나이", "직업"])
        writer.writerow(["김철수", "30", "개발자"])
        writer.writerow(["이영희", "25", "디자이너"])
        writer.writerow(["박민준", "35", "데이터 분석가"])

    # CSV 읽기 테스트
    result = read_file.invoke({"file_path": "sample.csv", "max_rows": 10})
    print(result)

    # 파일 정보 조회
    print("\n=== 파일 정보 ===")
    info = get_file_info.invoke({"file_path": "sample.csv"})
    print(info)

    # 임시 파일 정리
    os.unlink(temp_csv_path)
    print("\n[테스트 완료]")

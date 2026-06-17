"""
데이터 변환 도구 모듈

Pydantic 검증 기반으로 다양한 데이터 변환 작업을 수행해요.
정규화, 타입 변환, 포맷 변환 등을 안전하게 처리해요.

주요 기능:
    - 텍스트 정규화 (대소문자, 공백, 특수문자)
    - 날짜/시간 형식 변환
    - 숫자 단위 변환
    - 딕셔너리/리스트 구조 변환
"""

import re
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, validator
from langchain.tools import tool


# ===================== Pydantic 입력 스키마 정의 =====================

class TextNormalizeInput(BaseModel):
    """텍스트 정규화 입력 스키마예요."""

    text: str = Field(description="정규화할 입력 텍스트")
    case: Literal["upper", "lower", "title", "none"] = Field(
        default="none",
        description="대소문자 변환 방식: upper(대문자), lower(소문자), title(단어 첫글자 대문자), none(변환 없음)"
    )
    strip_whitespace: bool = Field(
        default=True,
        description="앞뒤 공백과 연속 공백을 제거할지 여부"
    )
    remove_special_chars: bool = Field(
        default=False,
        description="알파벳, 숫자, 한글, 기본 구두점을 제외한 특수문자 제거"
    )

    @validator("text")
    def text_must_not_be_empty(cls, v: str) -> str:
        """빈 텍스트를 허용하지 않아요."""
        if not v.strip():
            raise ValueError("텍스트가 비어 있어요")
        return v


class DateConvertInput(BaseModel):
    """날짜 변환 입력 스키마예요."""

    date_string: str = Field(
        description="변환할 날짜 문자열 (예: '2024-01-15', '15/01/2024', '20240115')"
    )
    output_format: str = Field(
        default="%Y년 %m월 %d일",
        description="출력 날짜 형식 (strftime 형식, 예: '%Y-%m-%d', '%Y년 %m월 %d일')"
    )


class UnitConvertInput(BaseModel):
    """단위 변환 입력 스키마예요."""

    value: float = Field(description="변환할 숫자 값")
    from_unit: str = Field(description="원본 단위 (예: km, kg, celsius, usd)")
    to_unit: str = Field(description="변환 목표 단위 (예: miles, lb, fahrenheit, krw)")


# ===================== 변환 함수 구현 =====================

def normalize_whitespace(text: str) -> str:
    """연속된 공백을 단일 공백으로 줄여요.

    Args:
        text: 정규화할 텍스트

    Returns:
        정규화된 텍스트
    """
    return re.sub(r"\s+", " ", text).strip()


def apply_case_transform(text: str, case: str) -> str:
    """텍스트에 대소문자 변환을 적용해요.

    Args:
        text: 변환할 텍스트
        case: 변환 방식 (upper/lower/title/none)

    Returns:
        변환된 텍스트
    """
    transforms = {
        "upper": str.upper,
        "lower": str.lower,
        "title": str.title,
        "none": lambda x: x
    }
    return transforms.get(case, lambda x: x)(text)


# 지원하는 날짜 입력 형식 목록
DATE_INPUT_FORMATS = [
    "%Y-%m-%d",    # 2024-01-15
    "%Y/%m/%d",    # 2024/01/15
    "%d/%m/%Y",    # 15/01/2024
    "%d-%m-%Y",    # 15-01-2024
    "%Y%m%d",      # 20240115
    "%m/%d/%Y",    # 01/15/2024
]

# 단위 변환 테이블 (기준 단위로의 변환 계수)
UNIT_CONVERSIONS: dict[str, dict[str, float]] = {
    # 길이 (기준: 미터)
    "km": {"meters": 1000, "miles": 0.621371, "feet": 3280.84, "cm": 100000},
    "miles": {"km": 1.60934, "meters": 1609.34, "feet": 5280, "cm": 160934},
    "meters": {"km": 0.001, "miles": 0.000621, "feet": 3.28084, "cm": 100},
    # 무게 (기준: kg)
    "kg": {"lb": 2.20462, "g": 1000, "oz": 35.274},
    "lb": {"kg": 0.453592, "g": 453.592, "oz": 16},
    "g": {"kg": 0.001, "lb": 0.00220462},
    # 온도 (별도 처리)
    "celsius": {"fahrenheit": None, "kelvin": None},
    "fahrenheit": {"celsius": None, "kelvin": None},
    # 통화 (예시 환율 - 실제 프로덕션에서는 실시간 API 사용)
    "usd": {"krw": 1350.0, "eur": 0.92, "jpy": 149.5},
    "krw": {"usd": 0.00074, "eur": 0.00068},
}


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """온도 단위를 변환해요.

    Args:
        value: 변환할 온도값
        from_unit: 원본 단위 (celsius/fahrenheit/kelvin)
        to_unit: 목표 단위

    Returns:
        변환된 온도값

    Raises:
        ValueError: 지원하지 않는 단위 조합
    """
    # 먼저 섭씨로 변환
    if from_unit == "celsius":
        celsius = value
    elif from_unit == "fahrenheit":
        celsius = (value - 32) * 5 / 9
    elif from_unit == "kelvin":
        celsius = value - 273.15
    else:
        raise ValueError(f"알 수 없는 온도 단위: {from_unit}")

    # 섭씨에서 목표 단위로 변환
    if to_unit == "celsius":
        return round(celsius, 2)
    elif to_unit == "fahrenheit":
        return round(celsius * 9 / 5 + 32, 2)
    elif to_unit == "kelvin":
        return round(celsius + 273.15, 2)
    else:
        raise ValueError(f"알 수 없는 온도 단위: {to_unit}")


# ===================== LangChain 도구 정의 =====================

@tool(args_schema=TextNormalizeInput)
def normalize_text(
    text: str,
    case: str = "none",
    strip_whitespace: bool = True,
    remove_special_chars: bool = False
) -> str:
    """텍스트를 정규화해요 (대소문자, 공백, 특수문자 처리).

    사용자 입력, 파일 내용, API 응답 텍스트를 표준 형식으로 변환해요.

    Args:
        text: 정규화할 텍스트
        case: 대소문자 변환 (upper/lower/title/none)
        strip_whitespace: 앞뒤 및 연속 공백 제거
        remove_special_chars: 특수문자 제거

    Returns:
        정규화된 텍스트
    """
    result = text

    if strip_whitespace:
        result = normalize_whitespace(result)

    if remove_special_chars:
        # 한글, 영문, 숫자, 기본 구두점만 유지
        result = re.sub(r"[^\w\s가-힣.,!?;:\-]", "", result)

    result = apply_case_transform(result, case)

    return result


@tool(args_schema=DateConvertInput)
def convert_date_format(
    date_string: str,
    output_format: str = "%Y년 %m월 %d일"
) -> str:
    """날짜 문자열을 원하는 형식으로 변환해요.

    다양한 입력 형식을 자동으로 인식하고 원하는 형식으로 변환해요.

    Args:
        date_string: 변환할 날짜 문자열
        output_format: 출력 형식 (strftime 형식)

    Returns:
        변환된 날짜 문자열
    """
    parsed_date = None

    for fmt in DATE_INPUT_FORMATS:
        try:
            parsed_date = datetime.strptime(date_string.strip(), fmt)
            break
        except ValueError:
            continue

    if parsed_date is None:
        supported = ", ".join(DATE_INPUT_FORMATS)
        return f"날짜 파싱 실패: '{date_string}'\n지원 형식: {supported}"

    try:
        return parsed_date.strftime(output_format)
    except Exception as e:
        return f"형식 오류: '{output_format}' - {e}"


@tool(args_schema=UnitConvertInput)
def convert_unit(value: float, from_unit: str, to_unit: str) -> str:
    """숫자 값의 단위를 변환해요.

    길이(km/miles/meters), 무게(kg/lb/g), 온도(celsius/fahrenheit),
    통화(usd/krw/eur) 변환을 지원해요.

    Args:
        value: 변환할 값
        from_unit: 원본 단위
        to_unit: 목표 단위

    Returns:
        변환 결과 문자열 (예: "100 km = 62.14 miles")
    """
    from_lower = from_unit.lower()
    to_lower = to_unit.lower()

    # 온도 변환은 별도 처리
    if from_lower in ("celsius", "fahrenheit", "kelvin"):
        try:
            result = convert_temperature(value, from_lower, to_lower)
            return f"{value} {from_unit} = {result} {to_unit}"
        except ValueError as e:
            return str(e)

    # 일반 단위 변환
    if from_lower not in UNIT_CONVERSIONS:
        return f"'{from_unit}' 단위를 지원하지 않아요"

    conversion_map = UNIT_CONVERSIONS[from_lower]

    if to_lower not in conversion_map:
        supported = ", ".join(conversion_map.keys())
        return f"'{from_unit}'에서 '{to_unit}'으로 변환할 수 없어요\n변환 가능 단위: {supported}"

    factor = conversion_map[to_lower]
    result = round(value * factor, 4)

    return f"{value} {from_unit} = {result:,} {to_unit}"


@tool
def flatten_dict(
    data: dict[str, Any],
    separator: str = ".",
    max_depth: int = 3
) -> str:
    """중첩된 딕셔너리를 평탄화해요 (점 표기법으로 키 생성).

    중첩된 JSON 구조를 플랫한 키-값 쌍으로 변환해요.
    설정 파일이나 API 응답 분석 시 유용해요.

    Args:
        data: 평탄화할 중첩 딕셔너리
        separator: 키 구분자 (기본: '.')
        max_depth: 최대 중첩 깊이 (기본: 3)

    Returns:
        평탄화된 키-값 쌍 목록 (문자열)
    """
    def _flatten(obj: Any, prefix: str, depth: int) -> dict[str, Any]:
        items: dict[str, Any] = {}

        if depth > max_depth:
            items[prefix] = str(obj)[:100]
            return items

        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}{separator}{k}" if prefix else k
                items.update(_flatten(v, new_key, depth + 1))
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:5]):  # 배열은 최대 5개만
                new_key = f"{prefix}[{i}]"
                items.update(_flatten(v, new_key, depth + 1))
            if len(obj) > 5:
                items[f"{prefix}[...]"] = f"(총 {len(obj)}개 중 5개 표시)"
        else:
            items[prefix] = obj

        return items

    flat = _flatten(data, "", 0)
    lines = [f"{k}: {v}" for k, v in flat.items()]
    return "\n".join(lines) if lines else "빈 딕셔너리예요"


if __name__ == "__main__":
    print("=== 데이터 변환기 테스트 ===")

    # 텍스트 정규화
    print("--- 텍스트 정규화 ---")
    result = normalize_text.invoke({
        "text": "  Hello   WORLD!  안녕하세요  ",
        "case": "title",
        "strip_whitespace": True
    })
    print(f"결과: {result}")

    # 날짜 변환
    print("\n--- 날짜 형식 변환 ---")
    for date_str in ["2024-01-15", "15/01/2024", "20240115"]:
        result = convert_date_format.invoke({
            "date_string": date_str,
            "output_format": "%Y년 %m월 %d일"
        })
        print(f"  {date_str} → {result}")

    # 단위 변환
    print("\n--- 단위 변환 ---")
    conversions = [
        {"value": 100, "from_unit": "km", "to_unit": "miles"},
        {"value": 70, "from_unit": "kg", "to_unit": "lb"},
        {"value": 25, "from_unit": "celsius", "to_unit": "fahrenheit"},
        {"value": 100, "from_unit": "usd", "to_unit": "krw"},
    ]
    for conv in conversions:
        result = convert_unit.invoke(conv)
        print(f"  {result}")

    # 딕셔너리 평탄화
    print("\n--- 딕셔너리 평탄화 ---")
    nested = {
        "user": {
            "name": "김철수",
            "address": {
                "city": "서울",
                "district": "강남구"
            }
        },
        "scores": [90, 85, 92]
    }
    result = flatten_dict.invoke({"data": nested, "separator": "."})
    print(result)

    print("\n[테스트 완료]")

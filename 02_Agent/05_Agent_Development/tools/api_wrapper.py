"""
외부 REST API 래핑 도구 모듈

외부 REST API를 LangChain 에이전트에서 사용할 수 있도록 래핑해요.
인증, 요청, 응답 파싱을 표준화된 방식으로 처리해요.

지원하는 공개 API (API 키 불필요):
    - Open-Meteo: 날씨 정보
    - JSONPlaceholder: 테스트용 CRUD API
    - CountriesNow: 국가/도시 정보
"""

from typing import Literal, Optional
from dataclasses import dataclass, field

import requests
from pydantic import BaseModel, Field
from langchain.tools import tool


# HTTP 메서드 타입 정의
HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]


@dataclass
class ApiResponse:
    """API 응답 데이터 클래스예요."""

    status_code: int
    data: dict | list | None
    error: Optional[str] = None
    headers: dict = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """2xx 상태 코드이면 성공이에요."""
        return 200 <= self.status_code < 300


def make_request(
    method: str,
    url: str,
    params: Optional[dict] = None,
    json_body: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = 15
) -> ApiResponse:
    """HTTP 요청을 수행하고 표준화된 응답을 반환해요.

    Args:
        method: HTTP 메서드 (GET, POST, PUT, DELETE, PATCH)
        url: 요청 URL
        params: 쿼리 파라미터 딕셔너리
        json_body: JSON 요청 바디
        headers: 추가 헤더
        timeout: 요청 타임아웃 (초)

    Returns:
        ApiResponse 객체
    """
    default_headers = {"Content-Type": "application/json"}
    if headers:
        default_headers.update(headers)

    allowed_hosts = {
        "api.open-meteo.com",
        "jsonplaceholder.typicode.com",
        "countriesnow.space",
        "api.ipify.org",
        "ipapi.co",
    }
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        return ApiResponse(
            status_code=0,
            data=None,
            error="허용된 공개 HTTPS API만 호출할 수 있어요."
        )

    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            params=params,
            json=json_body,
            headers=default_headers,
            timeout=timeout,
            allow_redirects=False,
        )

        # JSON 파싱 시도
        try:
            data = response.json()
        except ValueError:
            data = {"raw_text": response.text[:1000]}

        return ApiResponse(
            status_code=response.status_code,
            data=data,
            headers=dict(response.headers)
        )

    except requests.Timeout:
        return ApiResponse(
            status_code=0,
            data=None,
            error=f"요청 타임아웃: {timeout}초 초과"
        )
    except requests.ConnectionError:
        return ApiResponse(
            status_code=0,
            data=None,
            error="연결 오류: 서버에 연결할 수 없어요"
        )
    except Exception as e:
        return ApiResponse(
            status_code=0,
            data=None,
            error=f"예상치 못한 오류: {e}"
        )


class WeatherInput(BaseModel):
    """날씨 조회 입력 스키마예요."""

    latitude: float = Field(
        description="위도 (예: 서울=37.5665, 도쿄=35.6762)",
        ge=-90.0,
        le=90.0
    )
    longitude: float = Field(
        description="경도 (예: 서울=126.9780, 도쿄=139.6503)",
        ge=-180.0,
        le=180.0
    )
    days: int = Field(
        default=3,
        description="예보 기간 (1~7일, 기본값: 3일)",
        ge=1,
        le=7
    )


@tool(args_schema=WeatherInput)
def get_weather_forecast(
    latitude: float,
    longitude: float,
    days: int = 3
) -> str:
    """Open-Meteo API로 날씨 예보를 조회해요 (API 키 불필요).

    Args:
        latitude: 위도 (-90 ~ 90)
        longitude: 경도 (-180 ~ 180)
        days: 예보 기간 (1~7일)

    Returns:
        날짜별 최고/최저 온도와 강수량 예보
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "forecast_days": days,
        "timezone": "Asia/Seoul"
    }

    response = make_request("GET", url, params=params)

    if not response.is_success:
        return f"날씨 조회 실패 (상태 코드: {response.status_code}): {response.error or ''}"

    daily = response.data.get("daily", {})
    dates = daily.get("time", [])
    temp_max = daily.get("temperature_2m_max", [])
    temp_min = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])

    if not dates:
        return "날씨 데이터를 가져오지 못했어요"

    lines = [f"위도 {latitude}, 경도 {longitude} 기준 {days}일 예보:"]
    for i, date in enumerate(dates):
        t_max = temp_max[i] if i < len(temp_max) else "N/A"
        t_min = temp_min[i] if i < len(temp_min) else "N/A"
        rain = precip[i] if i < len(precip) else 0

        lines.append(
            f"  {date}: 최고 {t_max}°C / 최저 {t_min}°C, 강수량 {rain}mm"
        )

    return "\n".join(lines)


class RestApiInput(BaseModel):
    """범용 REST API 호출 입력 스키마예요."""

    url: str = Field(description="요청할 API 엔드포인트 URL")
    method: Literal["GET", "POST", "PUT", "DELETE"] = Field(
        default="GET",
        description="HTTP 메서드 (GET, POST, PUT, DELETE)"
    )
    params: Optional[dict] = Field(
        default=None,
        description="쿼리 파라미터 딕셔너리 (GET 요청에 사용)"
    )
    body: Optional[dict] = Field(
        default=None,
        description="JSON 요청 바디 (POST, PUT 요청에 사용)"
    )


@tool(args_schema=RestApiInput)
def call_rest_api(
    url: str,
    method: str = "GET",
    params: Optional[dict] = None,
    body: Optional[dict] = None
) -> str:
    """범용 REST API를 호출해요.

    학생용 교재에서는 미리 허용된 공개 HTTPS API만 호출할 수 있어요.

    Args:
        url: 요청할 API URL
        method: HTTP 메서드 (GET/POST/PUT/DELETE)
        params: 쿼리 파라미터
        body: JSON 요청 바디

    Returns:
        API 응답 데이터 (JSON 형태의 문자열)
    """
    import json as json_lib

    response = make_request(method, url, params=params, json_body=body)

    if response.error:
        return f"API 호출 오류: {response.error}"

    if not response.is_success:
        return f"API 오류 (상태 코드: {response.status_code}): {json_lib.dumps(response.data, ensure_ascii=False)[:500]}"

    # 응답 데이터를 보기 좋게 포맷
    formatted = json_lib.dumps(response.data, ensure_ascii=False, indent=2)

    # 너무 길면 잘라내기
    if len(formatted) > 2000:
        formatted = formatted[:2000] + "\n... (응답이 너무 길어 잘렸어요)"

    return f"상태 코드: {response.status_code}\n응답:\n{formatted}"


@tool
def get_public_ip() -> str:
    """현재 서버의 공개 IP 주소와 위치 정보를 조회해요.

    Returns:
        IP 주소, 국가, 도시 정보
    """
    response = make_request("GET", "https://ipapi.co/json/")

    if not response.is_success:
        return "IP 정보를 가져오지 못했어요"

    data = response.data
    return (
        f"IP 주소: {data.get('ip', 'N/A')}\n"
        f"국가: {data.get('country_name', 'N/A')} ({data.get('country_code', 'N/A')})\n"
        f"도시: {data.get('city', 'N/A')}\n"
        f"지역: {data.get('region', 'N/A')}\n"
        f"시간대: {data.get('timezone', 'N/A')}"
    )


if __name__ == "__main__":
    # 독립 실행 테스트: 서울 날씨 조회
    print("=== API 래퍼 테스트 ===")

    # 서울 날씨 조회 (위도: 37.5665, 경도: 126.9780)
    print("--- 서울 날씨 예보 (3일) ---")
    weather = get_weather_forecast.invoke({
        "latitude": 37.5665,
        "longitude": 126.9780,
        "days": 3
    })
    print(weather)

    # JSONPlaceholder API 테스트
    print("\n--- JSONPlaceholder GET 테스트 ---")
    result = call_rest_api.invoke({
        "url": "https://jsonplaceholder.typicode.com/posts/1",
        "method": "GET"
    })
    print(result)

    print("\n[테스트 완료]")

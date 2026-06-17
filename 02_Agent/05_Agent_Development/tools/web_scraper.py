"""
웹 스크래핑 도구 모듈

BeautifulSoup 기반으로 웹 페이지에서 텍스트, 링크, 메타데이터를 추출해요.
LangChain @tool 데코레이터와 함께 사용할 수 있는 에이전트용 도구 함수들을 제공해요.

의존성:
    pip install requests beautifulsoup4 langchain
"""

from typing import Optional
from dataclasses import dataclass, field
from ipaddress import ip_address
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool


@dataclass
class ScrapedPage:
    """스크래핑 결과 데이터 클래스예요."""

    url: str
    title: str
    text: str
    links: list[str] = field(default_factory=list)
    meta_description: str = ""


def fetch_page(url: str, timeout: int = 10) -> Optional[BeautifulSoup]:
    """URL에서 HTML을 가져와 BeautifulSoup 객체로 반환해요.

    Args:
        url: 스크래핑할 페이지 URL
        timeout: 요청 타임아웃 (초)

    Returns:
        BeautifulSoup 객체 또는 실패 시 None
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; LangGraphBot/1.0)"
        )
    }
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        print("HTTPS 공개 URL만 스크래핑할 수 있어요.")
        return None
    hostname = parsed.hostname.lower()
    if hostname in {"localhost", "0.0.0.0"} or hostname.endswith(".local"):
        print("로컬/내부 네트워크 주소는 허용하지 않아요.")
        return None
    try:
        host_ip = ip_address(hostname)
        if (
            host_ip.is_private
            or host_ip.is_loopback
            or host_ip.is_link_local
            or host_ip.is_multicast
            or host_ip.is_reserved
            or host_ip.is_unspecified
        ):
            print("로컬/내부 네트워크 주소는 허용하지 않아요.")
            return None
    except ValueError:
        try:
            resolved_ips = {
                item[4][0] for item in socket.getaddrinfo(hostname, None)
            }
        except socket.gaierror:
            print("호스트를 확인할 수 없어요.")
            return None
        for resolved in resolved_ips:
            resolved_ip = ip_address(resolved)
            if (
                resolved_ip.is_private
                or resolved_ip.is_loopback
                or resolved_ip.is_link_local
                or resolved_ip.is_multicast
                or resolved_ip.is_reserved
                or resolved_ip.is_unspecified
            ):
                print("로컬/내부 네트워크 주소는 허용하지 않아요.")
                return None

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=False,
        )
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except requests.RequestException as e:
        print(f"요청 실패: {e}")
        return None


def extract_main_text(soup: BeautifulSoup, max_length: int = 3000) -> str:
    """HTML에서 본문 텍스트를 추출해요.

    Args:
        soup: BeautifulSoup 파싱 객체
        max_length: 반환할 최대 텍스트 길이

    Returns:
        정제된 본문 텍스트
    """
    # 스크립트, 스타일 태그 제거
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # 빈 줄 정리
    lines = [line for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)

    return cleaned[:max_length] if len(cleaned) > max_length else cleaned


def extract_links(soup: BeautifulSoup, base_url: str, max_links: int = 20) -> list[str]:
    """페이지에서 절대 경로 링크를 추출해요.

    Args:
        soup: BeautifulSoup 파싱 객체
        base_url: 상대 경로를 절대 경로로 변환할 기준 URL
        max_links: 반환할 최대 링크 수

    Returns:
        절대 경로 URL 목록
    """
    from urllib.parse import urljoin, urlparse

    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        absolute = urljoin(base_url, href)
        # HTTP/HTTPS 링크만 수집
        if urlparse(absolute).scheme in ("http", "https"):
            links.append(absolute)
        if len(links) >= max_links:
            break
    return list(set(links))  # 중복 제거


@tool
def scrape_webpage(url: str, include_links: bool = False) -> str:
    """웹 페이지의 텍스트 내용을 스크래핑해요.

    주어진 URL의 웹 페이지에서 제목과 본문 텍스트를 추출해요.
    뉴스 기사, 블로그 포스트, 공식 문서 등의 내용을 읽을 때 사용해요.

    Args:
        url: 스크래핑할 웹 페이지 URL (https:// 로 시작해야 해요)
        include_links: True이면 페이지 내 링크 목록도 반환해요

    Returns:
        제목과 본문 텍스트 (include_links=True이면 링크 목록 포함)
    """
    soup = fetch_page(url)
    if soup is None:
        return f"'{url}' 페이지를 가져오는 데 실패했어요"

    # 제목 추출
    title = soup.title.string.strip() if soup.title else "제목 없음"

    # 본문 추출
    text = extract_main_text(soup)

    result = f"[제목] {title}\n\n[본문]\n{text}"

    if include_links:
        links = extract_links(soup, url)
        links_text = "\n".join(f"- {link}" for link in links[:10])
        result += f"\n\n[링크 목록]\n{links_text}"

    return result


@tool
def scrape_multiple_pages(urls: list[str]) -> str:
    """여러 웹 페이지를 순서대로 스크래핑해요.

    최대 5개의 URL을 받아 각 페이지의 내용을 요약해서 반환해요.
    비교 분석이나 여러 출처를 종합할 때 사용해요.

    Args:
        urls: 스크래핑할 URL 목록 (최대 5개)

    Returns:
        각 페이지의 제목과 요약 텍스트 (구분선으로 분리)
    """
    # 최대 5개로 제한
    target_urls = urls[:5]
    results = []

    for i, url in enumerate(target_urls, 1):
        soup = fetch_page(url)
        if soup is None:
            results.append(f"[{i}] {url}\n  - 로드 실패")
            continue

        title = soup.title.string.strip() if soup.title else "제목 없음"
        text = extract_main_text(soup, max_length=500)  # 요약용으로 500자로 제한

        results.append(f"[{i}] {title}\nURL: {url}\n{text}")

    return "\n\n" + "=" * 40 + "\n\n".join(results)


if __name__ == "__main__":
    # 독립 실행 테스트: 파이썬 공식 문서 스크래핑
    test_url = "https://www.python.org/"
    print("=== 웹 스크래퍼 테스트 ===")
    print(f"URL: {test_url}\n")

    result = scrape_webpage.invoke({"url": test_url, "include_links": False})
    # 처음 500자만 출력
    print(result[:500])
    print("\n[테스트 완료]")

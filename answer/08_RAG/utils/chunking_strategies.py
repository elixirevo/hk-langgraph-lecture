"""
청킹 전략 유틸리티 모듈

이 모듈은 다양한 텍스트 청킹 전략을 제공해요.
고정 크기, 재귀적, 시맨틱 청킹을 비교하고
최적의 청크 크기를 찾는 데 도움을 줘요.
"""

from typing import Callable, Optional

from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)


def fixed_size_chunking(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 0,
) -> list[Document]:
    """고정 크기 청킹 전략이에요.

    문서를 일정한 글자 수로 분할해요.
    단순하지만 문장 중간에서 잘릴 수 있어요.

    Args:
        documents: 분할할 Document 리스트
        chunk_size: 고정 청크 크기 (글자 수)
        chunk_overlap: 청크 간 겹치는 글자 수

    Returns:
        고정 크기로 분할된 Document 리스트
    """
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="",  # 구분자 없이 고정 크기로 분할
    )
    chunks = splitter.split_documents(documents)
    print(f"고정 크기 청킹: {len(documents)}개 → {len(chunks)}개")
    return chunks


def recursive_chunking(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[list[str]] = None,
) -> list[Document]:
    """재귀적 문자 기반 청킹 전략이에요.

    단락 → 문장 → 단어 순으로 자연스러운 경계에서 분할해요.
    문맥 보존이 가장 뛰어나 일반적으로 권장되는 전략이에요.

    Args:
        documents: 분할할 Document 리스트
        chunk_size: 최대 청크 크기 (글자 수)
        chunk_overlap: 청크 간 겹치는 글자 수
        separators: 분할 기준 문자 리스트 (None이면 기본값 사용)

    Returns:
        재귀적으로 분할된 Document 리스트
    """
    # 기본 구분자: 단락 → 줄바꿈 → 문장 → 단어
    if separators is None:
        separators = ["\n\n", "\n", ".", "!", "?", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    chunks = splitter.split_documents(documents)
    print(f"재귀 청킹: {len(documents)}개 → {len(chunks)}개")
    return chunks


def token_based_chunking(
    documents: list[Document],
    chunk_size: int = 256,
    chunk_overlap: int = 32,
    encoding_name: str = "cl100k_base",
) -> list[Document]:
    """토큰 기반 청킹 전략이에요.

    글자 수가 아닌 LLM 토큰 수를 기준으로 분할해요.
    컨텍스트 윈도우 제한이 있는 모델에 적합해요.

    Args:
        documents: 분할할 Document 리스트
        chunk_size: 최대 청크 토큰 수
        chunk_overlap: 청크 간 겹치는 토큰 수
        encoding_name: tiktoken 인코딩 이름

    Returns:
        토큰 기반으로 분할된 Document 리스트
    """
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name,
    )
    chunks = splitter.split_documents(documents)
    print(f"토큰 청킹: {len(documents)}개 → {len(chunks)}개")
    return chunks


def compare_strategies(
    documents: list[Document],
    chunk_size: int = 500,
) -> dict[str, list[Document]]:
    """세 가지 청킹 전략을 비교하는 함수예요.

    동일한 문서에 세 가지 전략을 적용하고
    결과를 딕셔너리로 반환해요.

    Args:
        documents: 분할할 Document 리스트
        chunk_size: 청크 크기 (세 전략에 공통 적용)

    Returns:
        전략명 → 청크 리스트 딕셔너리
    """
    print(f"=== 청킹 전략 비교 (chunk_size={chunk_size}) ===")

    results = {
        "고정_크기": fixed_size_chunking(documents, chunk_size=chunk_size),
        "재귀_문자": recursive_chunking(documents, chunk_size=chunk_size),
        "토큰_기반": token_based_chunking(
            documents,
            chunk_size=chunk_size // 3  # 토큰은 글자 수의 약 1/3
        ),
    }

    # 비교 통계 출력
    print("\n=== 비교 결과 ===")
    for strategy_name, chunks in results.items():
        avg_len = sum(len(c.page_content) for c in chunks) / max(len(chunks), 1)
        print(f"  {strategy_name}: {len(chunks)}개 청크, 평균 {avg_len:.0f}자")

    return results


def calculate_chunk_stats(chunks: list[Document]) -> dict[str, float]:
    """청크 리스트의 통계를 계산해요.

    Args:
        chunks: Document 리스트

    Returns:
        통계 딕셔너리 (count, avg_length, min_length, max_length)
    """
    if not chunks:
        return {"count": 0, "avg_length": 0, "min_length": 0, "max_length": 0}

    lengths = [len(c.page_content) for c in chunks]

    return {
        "count": len(chunks),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }


if __name__ == "__main__":
    # 독립 실행 테스트
    import os
    import sys

    # 스크립트 위치 기준으로 data 디렉토리 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    pdf_path = os.path.join(data_dir, "SPRI_AI_Brief_2023년12월호_F.pdf")

    if not os.path.exists(pdf_path):
        print(f"오류: {pdf_path} 파일을 찾을 수 없어요")
        print("data/ 디렉토리에 SPRI_AI_Brief_2023년12월호_F.pdf를 넣어주세요")
        sys.exit(1)

    from langchain_community.document_loaders import PyPDFLoader

    print("=== 청킹 전략 비교 테스트 ===")
    pages = PyPDFLoader(pdf_path).load()
    print(f"PDF 로드 완료: {len(pages)}페이지")

    # 처음 5페이지만 테스트에 사용
    test_docs = pages[:5]

    # 전략 비교
    comparison = compare_strategies(test_docs, chunk_size=300)

    # 각 전략의 첫 청크 미리보기
    print("\n=== 첫 청크 미리보기 ===")
    for strategy_name, chunks in comparison.items():
        if chunks:
            print(f"\n[{strategy_name}]")
            print(chunks[0].page_content[:150] + "...")

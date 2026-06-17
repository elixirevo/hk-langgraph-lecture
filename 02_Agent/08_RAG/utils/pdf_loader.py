"""
PDF 로더 유틸리티 모듈

이 모듈은 다양한 소스의 PDF 문서를 로드하고,
ChromaDB 벡터스토어로 변환하는 기능을 제공해요.
단일 파일부터 디렉토리 전체 로드까지 지원해요.
"""

from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf(file_path: str) -> list[Document]:
    """단일 PDF 파일을 페이지 단위로 로드해요.

    Args:
        file_path: PDF 파일 경로

    Returns:
        페이지 단위로 분할된 Document 리스트
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"로드 완료: {file_path} ({len(pages)}페이지)")
    return pages


def load_pdfs_from_directory(directory: str, pattern: str = "*.pdf") -> list[Document]:
    """디렉토리에서 모든 PDF 파일을 로드해요.

    Args:
        directory: 디렉토리 경로
        pattern: 파일 검색 패턴 (기본: 모든 PDF)

    Returns:
        모든 PDF의 Document 리스트 (파일별 메타데이터 포함)
    """
    dir_path = Path(directory)
    pdf_files = list(dir_path.glob(pattern))

    if not pdf_files:
        print(f"경고: {directory}에 PDF 파일이 없어요")
        return []

    all_docs: list[Document] = []
    for pdf_file in pdf_files:
        docs = load_pdf(str(pdf_file))
        # 각 문서에 파일명 메타데이터 추가
        for doc in docs:
            doc.metadata["file_name"] = pdf_file.name
        all_docs.extend(docs)

    print(f"\n총 {len(pdf_files)}개 파일, {len(all_docs)}페이지 로드 완료")
    return all_docs


def create_vectorstore(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model: str = "text-embedding-3-small",
) -> Chroma:
    """문서 리스트를 ChromaDB 벡터스토어로 변환해요.

    Args:
        documents: Document 리스트 (페이지 단위)
        chunk_size: 청크 최대 글자 수 (기본: 500)
        chunk_overlap: 청크 간 겹치는 글자 수 (기본: 50)
        embedding_model: OpenAI 임베딩 모델명

    Returns:
        검색 가능한 ChromaDB 벡터스토어
    """
    # 청킹: 긴 페이지를 작은 단위로 분할해요
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"청킹 완료: {len(documents)}페이지 → {len(chunks)}개 청크")

    # 임베딩 + 인덱스 생성
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_collection",
    )
    print(f"ChromaDB 컬렉션 생성 완료: {len(chunks)}개 벡터")

    return vectorstore


def load_and_index_pdf(
    file_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    save_path: Optional[str] = None,
) -> Chroma:
    """PDF 로드부터 ChromaDB 인덱스 생성까지 한 번에 처리해요.

    Args:
        file_path: PDF 파일 경로
        chunk_size: 청크 최대 글자 수
        chunk_overlap: 청크 간 겹치는 글자 수
        save_path: 인덱스 저장 경로 (ChromaDB는 persist_directory로 영속화, None이면 인메모리)

    Returns:
        검색 가능한 ChromaDB 벡터스토어
    """
    # 1. PDF 로드
    documents = load_pdf(file_path)

    # 2. 벡터스토어 생성
    vectorstore = create_vectorstore(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 3. 선택적 저장 (ChromaDB persist_directory 방식)
    if save_path:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        Chroma.from_documents(
            documents=vectorstore.get()["documents"] if hasattr(vectorstore, "get") else [],
            embedding=embeddings,
            collection_name="rag_collection",
            persist_directory=save_path,
        )
        print(f"ChromaDB 저장 완료: {save_path}")

    return vectorstore


if __name__ == "__main__":
    # 독립 실행 테스트
    # 현재 디렉토리에 data/SPRI_AI_Brief_2023년12월호_F.pdf 파일이 있어야 해요
    import sys
    import os

    # 스크립트 위치 기준으로 data 디렉토리 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    pdf_path = os.path.join(data_dir, "SPRI_AI_Brief_2023년12월호_F.pdf")

    if not os.path.exists(pdf_path):
        print(f"오류: {pdf_path} 파일을 찾을 수 없어요")
        sys.exit(1)

    print("=== PDF 로더 테스트 ===")
    vs = load_and_index_pdf(pdf_path)

    # 검색 테스트
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke("삼성전자 AI")
    print(f"\n검색 결과 ({len(results)}개):")
    for doc in results:
        print(f"  - 페이지 {doc.metadata.get('page', 0)+1}: {doc.page_content[:80]}...")

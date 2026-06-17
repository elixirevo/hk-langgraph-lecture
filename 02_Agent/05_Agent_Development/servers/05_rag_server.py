"""
RAG MCP 서버 (Retrieval-Augmented Generation MCP Server)

이 모듈은 PDF 문서에서 정보를 검색하는 RAG 기능을 MCP 서버로 제공해요.
FastMCP + FAISS + OpenAI 임베딩을 사용하여 의미 기반 검색을 구현해요.

주요 기능:
- PDF 문서 로드 및 벡터 DB 구축 (FAISS)
- 쿼리 기반 의미론적 문서 검색
- 검색 결과 포맷팅 및 반환

교육 목적: RAG 기능을 MCP 서버로 분리하면 에이전트 코드와 검색 로직을
           독립적으로 관리할 수 있어요. 검색 엔진 교체 시 에이전트 코드
           수정이 필요 없어요.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# 환경 변수를 로드해요 (OPENAI_API_KEY 필요)
load_dotenv(override=True)

# ---------------------------------------------------
# FastMCP 서버 초기화
# ---------------------------------------------------
mcp = FastMCP(
    "RAGRetriever",
    instructions="PDF 문서 데이터베이스에서 관련 정보를 검색하는 어시스턴트예요. retrieve 도구를 사용해 질문과 관련된 문서 내용을 찾아드려요.",
)

# ---------------------------------------------------
# 전역 벡터 스토어 (서버 시작 시 한 번만 초기화)
# ---------------------------------------------------
# 서버가 시작될 때 벡터 DB를 생성하고 재사용해요
_vector_store = None


def get_pdf_path() -> Optional[Path]:
    """PDF 파일 경로를 반환해요.

    서버 파일과 같은 디렉토리의 data/ 폴더에서 PDF를 찾아요.

    Returns:
        PDF 파일 경로 (없으면 None)
    """
    # 이 파일의 디렉토리를 기준으로 data/ 폴더를 찾아요
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"

    # data 폴더에서 첫 번째 PDF 파일을 반환해요
    pdf_files = list(data_dir.glob("*.pdf")) if data_dir.exists() else []

    if pdf_files:
        return pdf_files[0]

    # 상위 디렉토리의 data/ 폴더도 확인해요
    parent_data_dir = current_dir.parent / "data"
    pdf_files = list(parent_data_dir.glob("*.pdf")) if parent_data_dir.exists() else []

    return pdf_files[0] if pdf_files else None


def build_vector_store(pdf_path: Path):
    """PDF 문서로 FAISS 벡터 스토어를 구축해요.

    Args:
        pdf_path: PDF 파일 경로

    Returns:
        FAISS 벡터 스토어 객체
    """
    # 필요한 패키지를 임포트해요
    # 사용 시점에 임포트하여 서버 시작 시간을 줄여요
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print(f"[RAG] PDF 로드 중: {pdf_path}")

    # PDF 문서를 로드해요
    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()

    print(f"[RAG] 문서 로드 완료: {len(documents)}페이지")

    # 청크 크기 500, 오버랩 50으로 문서를 분할해요
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 청크당 최대 500자
        chunk_overlap=50,  # 인접 청크 간 50자 오버랩으로 문맥 유지
    )
    chunks = text_splitter.split_documents(documents)

    print(f"[RAG] 청크 생성 완료: {len(chunks)}개")

    # OpenAI 임베딩으로 벡터 DB를 구축해요
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    print("[RAG] 벡터 DB 구축 완료!")

    return vector_store


def get_vector_store():
    """벡터 스토어를 반환해요. 없으면 초기화 메시지를 출력해요.

    Returns:
        FAISS 벡터 스토어 또는 None
    """
    global _vector_store

    # 이미 초기화된 경우 재사용해요 (싱글톤 패턴)
    if _vector_store is not None:
        return _vector_store

    # PDF 파일을 찾아요
    pdf_path = get_pdf_path()

    if pdf_path is None:
        print("[RAG] 경고: PDF 파일을 찾을 수 없어요. servers/data/ 폴더에 PDF를 추가해주세요.")
        return None

    # 벡터 스토어를 구축해요
    _vector_store = build_vector_store(pdf_path)

    return _vector_store


# ---------------------------------------------------
# MCP 도구 정의
# ---------------------------------------------------


@mcp.tool()
async def retrieve(query: str, k: int = 4) -> str:
    """쿼리와 가장 관련있는 문서 청크를 검색하여 반환해요.

    FAISS 벡터 DB에서 코사인 유사도 기반으로 상위 k개의 문서를 검색해요.

    Args:
        query: 검색할 질문이나 키워드
        k: 반환할 최대 문서 수 (기본값: 4)

    Returns:
        관련 문서 내용을 연결한 문자열
    """
    # 벡터 스토어가 준비될 때까지 기다려요
    vector_store = get_vector_store()

    if vector_store is None:
        return (
            "RAG 서버를 사용할 수 없어요: PDF 파일이 없습니다. "
            "servers/data/ 폴더에 PDF 파일을 추가한 후 서버를 재시작해주세요."
        )

    try:
        # 유사도 검색으로 관련 문서를 가져와요
        docs = vector_store.similarity_search(query, k=k)

        if not docs:
            return f"'{query}'와 관련된 문서를 찾을 수 없었어요."

        # 검색된 문서 내용을 연결해요
        # 각 청크 사이에 구분선을 넣어 가독성을 높여요
        result_parts = []
        for i, doc in enumerate(docs, 1):
            source_info = ""
            if doc.metadata.get("source"):
                # 파일명만 추출하여 출처를 표시해요
                source_file = Path(doc.metadata["source"]).name
                page_num = doc.metadata.get("page", "?")
                source_info = f"[출처: {source_file}, {page_num}페이지] "

            result_parts.append(f"{source_info}{doc.page_content}")

        return "\n\n".join(result_parts)

    except Exception as e:
        return f"검색 오류: {str(e)}"


@mcp.tool()
async def get_retriever_status() -> str:
    """RAG 서버의 상태와 로드된 문서 정보를 반환해요.

    Returns:
        서버 상태 정보 문자열
    """
    vector_store = get_vector_store()

    if vector_store is None:
        pdf_path = get_pdf_path()
        if pdf_path is None:
            return "상태: 비활성\n이유: PDF 파일을 찾을 수 없어요. servers/data/ 폴더에 PDF를 추가해주세요."
        return f"상태: 초기화 중\nPDF 경로: {pdf_path}"

    # 벡터 스토어의 문서 수를 확인해요
    try:
        doc_count = vector_store.index.ntotal
        pdf_path = get_pdf_path()
        return (
            f"상태: 활성\n"
            f"PDF 파일: {pdf_path.name if pdf_path else '알 수 없음'}\n"
            f"벡터 DB 크기: {doc_count}개 벡터\n"
            f"사용 가능한 도구: retrieve"
        )
    except Exception:
        return "상태: 활성 (세부 정보 조회 불가)"


if __name__ == "__main__":
    print("RAG MCP 서버 시작 중...")
    print("사용 가능한 도구: retrieve, get_retriever_status")

    # 서버 시작 전 벡터 스토어를 미리 초기화해요
    # 첫 번째 검색 요청 시 지연을 줄여줘요
    print("\n벡터 DB 사전 초기화 중...")
    get_vector_store()

    print("\n서버 준비 완료! stdio 전송 방식으로 대기 중...")
    # stdio 전송 방식으로 MCP 서버를 시작해요
    mcp.run(transport="stdio")

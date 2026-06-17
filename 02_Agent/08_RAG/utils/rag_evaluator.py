"""
RAG 평가 유틸리티 모듈

이 모듈은 RAG 시스템의 품질을 평가하는 기능을 제공해요.
문서 관련성, 답변 충실도, 답변 관련성을 LLM 기반으로 평가해요.
단순한 이진 평가부터 점수 기반 평가까지 지원해요.
"""

from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from pydantic import BaseModel, Field


# 평가에 사용할 모델 (비용 효율)
EVAL_MODEL = "openai:gpt-4o-mini"


# ---------------------------------------------------
# 평가 결과 데이터 모델
# ---------------------------------------------------

class BinaryScore(BaseModel):
    """이진(yes/no) 평가 결과를 담는 모델이에요."""

    binary_score: Literal["yes", "no"] = Field(
        description="평가 결과: 'yes'(적합) 또는 'no'(부적합)"
    )
    reason: str = Field(
        description="평가 근거를 한 문장으로 설명해요"
    )


class NumericScore(BaseModel):
    """0-10 점수 기반 평가 결과를 담는 모델이에요."""

    score: int = Field(
        ge=0, le=10,
        description="평가 점수 (0=최저, 10=최고)"
    )
    reason: str = Field(
        description="점수를 부여한 근거를 설명해요"
    )


# ---------------------------------------------------
# 평가기 함수들
# ---------------------------------------------------

def evaluate_retrieval_relevance(
    question: str,
    document: Document | str,
    model_name: str = EVAL_MODEL,
) -> BinaryScore:
    """검색된 문서가 질문과 관련이 있는지 평가해요.

    RAG 파이프라인에서 불필요한 문서를 필터링하는 데 사용해요.

    Args:
        question: 사용자 질문
        document: 평가할 문서 (Document 객체 또는 텍스트)
        model_name: 평가에 사용할 LLM 모델명

    Returns:
        BinaryScore (yes=관련있음, no=관련없음)
    """
    # Document 객체이면 page_content 추출
    doc_text = document.page_content if isinstance(document, Document) else document

    llm = init_chat_model(model_name, temperature=0)
    structured_llm = llm.with_structured_output(BinaryScore)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a retrieval relevance evaluator.
Assess whether the retrieved document is relevant to the user question.
Score 'yes' if the document contains useful information for answering the question,
'no' otherwise. Provide a brief reason in Korean."""),
        ("human", "Question: {question}\n\nDocument:\n{document}"),
    ])

    chain = prompt | structured_llm
    result = chain.invoke({"question": question, "document": doc_text})
    return result


def evaluate_answer_faithfulness(
    answer: str,
    context_docs: list[Document] | list[str],
    model_name: str = EVAL_MODEL,
) -> BinaryScore:
    """생성된 답변이 검색 문서에 근거하는지 평가해요.

    환각(Hallucination) 여부를 검증하는 데 사용해요.
    yes = 문서에 근거함 (환각 없음)
    no = 문서에 없는 내용 포함 (환각 가능성)

    Args:
        answer: 평가할 생성 답변
        context_docs: 참고 문서 리스트
        model_name: 평가에 사용할 LLM 모델명

    Returns:
        BinaryScore (yes=근거있음, no=환각가능성)
    """
    # 문서 텍스트 추출 및 합치기
    if context_docs and isinstance(context_docs[0], Document):
        context_text = "\n\n".join([d.page_content for d in context_docs])
    else:
        context_text = "\n\n".join(context_docs)  # type: ignore

    llm = init_chat_model(model_name, temperature=0)
    structured_llm = llm.with_structured_output(BinaryScore)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a faithfulness evaluator.
Check whether the given answer is grounded in and supported by the provided context documents.
Score 'yes' if the answer is fully supported by the context,
'no' if it contains information not found in the context.
Provide a brief reason in Korean."""),
        ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
    ])

    chain = prompt | structured_llm
    result = chain.invoke({"context": context_text, "answer": answer})
    return result


def evaluate_answer_relevance(
    question: str,
    answer: str,
    model_name: str = EVAL_MODEL,
) -> BinaryScore:
    """생성된 답변이 실제로 질문에 답하는지 평가해요.

    답변이 질문의 의도를 충족하는지 검증해요.

    Args:
        question: 원본 사용자 질문
        answer: 평가할 생성 답변
        model_name: 평가에 사용할 LLM 모델명

    Returns:
        BinaryScore (yes=질문해결, no=질문미해결)
    """
    llm = init_chat_model(model_name, temperature=0)
    structured_llm = llm.with_structured_output(BinaryScore)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an answer relevance evaluator.
Assess whether the given answer adequately addresses the user's question.
Score 'yes' if the answer resolves the question,
'no' if it's off-topic or doesn't address the question.
Provide a brief reason in Korean."""),
        ("human", "Question: {question}\n\nAnswer: {answer}"),
    ])

    chain = prompt | structured_llm
    result = chain.invoke({"question": question, "answer": answer})
    return result


def evaluate_answer_quality(
    question: str,
    answer: str,
    model_name: str = EVAL_MODEL,
) -> NumericScore:
    """생성된 답변의 종합 품질을 0-10점으로 평가해요.

    정확성, 완성도, 가독성을 종합하여 평가해요.

    Args:
        question: 원본 사용자 질문
        answer: 평가할 생성 답변
        model_name: 평가에 사용할 LLM 모델명

    Returns:
        NumericScore (0-10점 + 근거)
    """
    llm = init_chat_model(model_name, temperature=0)
    structured_llm = llm.with_structured_output(NumericScore)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a comprehensive answer quality evaluator.
Rate the answer quality from 0 to 10 based on:
- Accuracy (정확성): Does the answer correctly address the question?
- Completeness (완성도): Does it cover all aspects of the question?
- Clarity (명확성): Is it well-organized and easy to understand?
Provide a score and brief reason in Korean."""),
        ("human", "Question: {question}\n\nAnswer: {answer}"),
    ])

    chain = prompt | structured_llm
    result = chain.invoke({"question": question, "answer": answer})
    return result


def run_full_evaluation(
    question: str,
    retrieved_docs: list[Document],
    answer: str,
    verbose: bool = True,
) -> dict[str, BinaryScore | NumericScore]:
    """RAG 파이프라인의 전체 평가를 실행해요.

    문서 관련성, 답변 충실도, 답변 관련성, 종합 품질을 평가해요.

    Args:
        question: 사용자 질문
        retrieved_docs: 검색된 문서 리스트
        answer: 생성된 답변
        verbose: 평가 결과를 즉시 출력할지 여부

    Returns:
        평가 결과 딕셔너리
    """
    results = {}

    # 1. 문서 관련성 평가 (첫 번째 문서만 샘플 평가)
    if retrieved_docs:
        relevance = evaluate_retrieval_relevance(question, retrieved_docs[0])
        results["retrieval_relevance"] = relevance
        if verbose:
            print(f"[문서 관련성] {relevance.binary_score} - {relevance.reason}")

    # 2. 답변 충실도 평가 (환각 검증)
    faithfulness = evaluate_answer_faithfulness(answer, retrieved_docs)
    results["faithfulness"] = faithfulness
    if verbose:
        print(f"[답변 충실도] {faithfulness.binary_score} - {faithfulness.reason}")

    # 3. 답변 관련성 평가
    answer_relevance = evaluate_answer_relevance(question, answer)
    results["answer_relevance"] = answer_relevance
    if verbose:
        print(f"[답변 관련성] {answer_relevance.binary_score} - {answer_relevance.reason}")

    # 4. 종합 품질 평가
    quality = evaluate_answer_quality(question, answer)
    results["quality_score"] = quality
    if verbose:
        print(f"[종합 품질] {quality.score}/10 - {quality.reason}")

    return results


if __name__ == "__main__":
    # 독립 실행 테스트
    import os
    import sys

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    pdf_path = os.path.join(data_dir, "SPRI_AI_Brief_2023년12월호_F.pdf")

    if not os.path.exists(pdf_path):
        print(f"오류: {pdf_path} 파일을 찾을 수 없어요")
        sys.exit(1)

    print("=== RAG 평가기 테스트 ===")

    # 샘플 데이터 생성
    sample_question = "삼성전자가 개발한 생성형 AI의 이름은?"
    sample_answer = "삼성전자가 개발한 생성형 AI의 이름은 '삼성 가우스'입니다."
    sample_doc = Document(
        page_content="삼성전자가 자체 개발한 생성 AI 모델 '삼성 가우스'를 공개했습니다.",
        metadata={"source": "test", "page": 9}
    )

    print(f"질문: {sample_question}")
    print(f"답변: {sample_answer}")
    print()

    # 전체 평가 실행
    results = run_full_evaluation(
        question=sample_question,
        retrieved_docs=[sample_doc],
        answer=sample_answer,
        verbose=True,
    )

    print(f"\n평가 완료: {len(results)}개 항목 평가됨")

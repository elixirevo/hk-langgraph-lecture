import uuid

import streamlit as st
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# LangGraph & langchain 관련 임포트
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# 세션 상태 초기화
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

# 스레드별 장바구니 글로벌 격리 저장소
if "GLOBAL_CARTS" not in st.session_state:
    st.session_state["GLOBAL_CARTS"] = {}

GLOBAL_CARTS = st.session_state["GLOBAL_CARTS"]

# Streamlit UI 설정
st.set_page_config(page_title="쇼핑 에이전트 🤖", page_icon="🛍️", layout="wide")

# 사이드바 설정
st.sidebar.title("🛍️ 쇼핑 도우미 설정")
st.sidebar.markdown("---")
ORDER_THRESHOLD = st.sidebar.slider(
    "고액 주문 기준 설정 (원)", 10000, 200000, 50000, step=5000
)
st.sidebar.write(f"현재 기준: **{ORDER_THRESHOLD:,}원**")
st.sidebar.markdown("> 이 금액 이상 주문 시 자동 거부됩니다.")

# 현재 사용자 세션의 장바구니 가져오기
current_cart = GLOBAL_CARTS.setdefault(st.session_state.thread_id, {})

# 장바구니 요약 정보 사이드바 표시
st.sidebar.subheader("🛒 현재 장바구니")
if not current_cart:
    st.sidebar.info("장바구니가 비어 있습니다.")
else:
    total_cart_amount = 0
    for prod, details in current_cart.items():
        subtotal = details["quantity"] * details["price"]
        total_cart_amount += subtotal
        st.sidebar.write(f"- **{prod}** x {details['quantity']} = {subtotal:,}원")
    st.sidebar.markdown(f"**총 예상 금액:** `{total_cart_amount:,}원`")

if st.sidebar.button("장바구니 비우기"):
    GLOBAL_CARTS[st.session_state.thread_id] = {}
    st.rerun()


# 도구 정의 - RunnableConfig를 통해 thread_id를 추적하여 글로벌 저장소에 기록
@tool
def add_to_cart(
    product_name: str, quantity: int, price_per_unit: int, config: RunnableConfig
) -> str:
    """장바구니에 상품을 추가합니다."""
    thread_id = config["configurable"]["thread_id"]
    user_cart = GLOBAL_CARTS.setdefault(thread_id, {})

    if product_name in user_cart:
        user_cart[product_name]["quantity"] += quantity
    else:
        user_cart[product_name] = {"quantity": quantity, "price": price_per_unit}

    total = quantity * price_per_unit
    return f"장바구니 추가 성공: {product_name} x{quantity} = {total:,}원"


@tool
def place_order(
    product_name: str, quantity: int, total_amount: int, config: RunnableConfig
) -> str:
    """주문을 확정합니다. 결제를 처리하기 위해 총 결제금액 정보를 포함합니다.
    만약 사용자가 상품의 개당 가격을 제공하지 않았다면, 기본 가격을 개당 10,000원으로 간주하여 총 금액(total_amount = quantity * 10000)을 직접 계산한 후 이 도구를 즉시 호출하세요.
    """
    thread_id = config["configurable"]["thread_id"]
    user_cart = GLOBAL_CARTS.setdefault(thread_id, {})

    if product_name in user_cart:
        del user_cart[product_name]
    return f"주문 완료: {product_name} x{quantity}, 결제 금액: {total_amount:,}원"


@tool
def cancel_order(order_id: str, reason: str) -> str:
    """주문을 취소합니다."""
    return f"주문 {order_id} 취소 완료. 사유: {reason}"


# 에이전트 생성
@st.cache_resource
def get_shopping_agent():
    model = init_chat_model("openai:gpt-4o-mini")
    agent = create_agent(
        model=model,
        tools=[add_to_cart, place_order, cancel_order],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "add_to_cart": False,
                    "place_order": {"allowed_decisions": ["approve", "reject"]},
                    "cancel_order": True,
                },
                description_prefix="[쇼핑 승인 대기]",
            )
        ],
        checkpointer=InMemorySaver(),
    )
    return agent


shopping_agent = get_shopping_agent()
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# 메인 UI
st.title("🤖 쇼핑 에이전트 (HITL)")
st.write(
    "LangGraph HITL 미들웨어 탑재 쇼핑 챗봇입니다. 주문 확정 금액에 따라 승인 대기/거부 정책이 동적으로 작동합니다."
)

# 대화 기록 렌더링
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 사용자의 입력 처리 함수
def handle_agent_invocation(user_msg):
    # 사용자 입력 추가
    st.session_state.messages.append({"role": "user", "content": user_msg})

    # 에이전트 호출
    state = {"messages": [{"role": "user", "content": user_msg}]}
    res = shopping_agent.invoke(state, config=config)
    process_agent_result(res)


# 에이전트 응답 처리 함수
def process_agent_result(res):
    # 만약 인터럽트가 있으면 세션 상태에 저장하여 UI에 노출
    if "__interrupt__" in res:
        interrupt_list = res["__interrupt__"]
        interrupt_id = interrupt_list[0].id
        interrupt_value = interrupt_list[0].value
        action_requests = interrupt_value.get("action_requests", [])

        # place_order의 경우 자동/수동 처리 분기
        for action in action_requests:
            if action.get("name") == "place_order":
                args = action.get("args", {})
                total_amount = args.get("total_amount", 0)

                # 5만원 미만인 경우 ➔ 자동 승인(approve) 처리
                if total_amount < ORDER_THRESHOLD:
                    st.toast(f"자동 승인됨: {total_amount:,}원 < {ORDER_THRESHOLD:,}원")
                    decisions = [{"type": "approve"}]
                    next_res = shopping_agent.invoke(
                        Command(resume={interrupt_id: {"decisions": decisions}}),
                        config=config,
                    )
                    process_agent_result(next_res)
                    return
                # 5만원 이상인 경우 ➔ 자동 거절(reject) 처리
                else:
                    st.toast(
                        f"자동 거절됨: {total_amount:,}원 >= {ORDER_THRESHOLD:,}원 (고액 주문)"
                    )
                    decisions = [
                        {
                            "type": "reject",
                            "message": f"주문 금액 {total_amount:,}원이 {ORDER_THRESHOLD:,}원 이상이므로 고액 주문으로 판단되어 자동으로 거부되었습니다.",
                        }
                    ]
                    next_res = shopping_agent.invoke(
                        Command(resume={interrupt_id: {"decisions": decisions}}),
                        config=config,
                    )
                    process_agent_result(next_res)
                    return
            elif action.get("name") == "cancel_order":
                # 주문 취소의 경우 대기 상태로 설정
                st.session_state.pending_interrupt = {
                    "id": interrupt_id,
                    "action": action,
                }
                st.rerun()
                return

    # 인터럽트가 없거나 처리가 끝났으면 최종 메시지 기록
    if "messages" in res and res["messages"]:
        last_msg = res["messages"][-1].content
        st.session_state.messages.append({"role": "assistant", "content": last_msg})
        st.rerun()


# 챗 인풋
if prompt := st.chat_input(
    "장바구니 추가나 주문을 요청해보세요! (예: '사과 10개 장바구니에 담고 주문해줘. 개당 3000원이야.')"
):
    # 이전 인터럽트가 펜딩 중이면 새로운 입력 불가
    if st.session_state.pending_interrupt:
        st.warning(
            "⚠️ 승인 대기 중인 주문이 있습니다. 먼저 승인 혹은 거부 결정을 내려주세요."
        )
    else:
        # 화면 즉시 리렌더링을 위해 메시지 임시 렌더링 후 처리
        with st.chat_message("user"):
            st.markdown(prompt)
        handle_agent_invocation(prompt)

# 승인 대기 UI 렌더링 (펜딩 중인 인터럽트가 있을 때)
if st.session_state.pending_interrupt:
    pending = st.session_state.pending_interrupt
    action = pending["action"]
    action_name = action.get("name")
    action_args = action.get("args", {})

    st.info(
        f"🔔 **[승인 대기]** 에이전트가 **{action_name}** 도구 실행을 위한 승인을 기다리고 있습니다."
    )

    if action_name == "cancel_order":
        st.warning(
            f"🚫 **주문 취소 확인**\n\n"
            f"- **주문 ID**: {action_args.get('order_id')}\n"
            f"- **취소 사유**: {action_args.get('reason')}"
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("👍 취소 승인 (Approve)", use_container_width=True):
                decisions = [{"type": "approve"}]
                st.session_state.pending_interrupt = None
                res = shopping_agent.invoke(
                    Command(resume={pending["id"]: {"decisions": decisions}}),
                    config=config,
                )
                process_agent_result(res)
        with col2:
            if st.button("👎 취소 거부 (Reject)", use_container_width=True):
                decisions = [
                    {"type": "reject", "message": "주문 취소 요청이 거절되었습니다."}
                ]
                st.session_state.pending_interrupt = None
                res = shopping_agent.invoke(
                    Command(resume={pending["id"]: {"decisions": decisions}}),
                    config=config,
                )
                process_agent_result(res)
        with col3:
            # 직접 응답(Respond) 예시
            if st.button("✉️ 직접 메시지 전달 (Respond)", use_container_width=True):
                decisions = [
                    {
                        "type": "respond",
                        "message": "주문 취소는 고객센터를 통해 직접 문의하시기 바랍니다.",
                    }
                ]
                st.session_state.pending_interrupt = None
                res = shopping_agent.invoke(
                    Command(resume={pending["id"]: {"decisions": decisions}}),
                    config=config,
                )
                process_agent_result(res)

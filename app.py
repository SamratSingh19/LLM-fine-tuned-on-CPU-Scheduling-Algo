import streamlit as st
import matplotlib.pyplot as plt
from inference.pipeline import answer_query
from engine.scheduler import Process, ALGORITHMS

st.set_page_config(page_title="SchedTutor", layout="centered")

st.markdown("""
    <style>
        header[data-testid="stHeader"] {display: none;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("LLM Fine-Tuned on CPU Scheduling Algorithms")

tab1, tab2 = st.tabs(["Ask a question", "Solve a numerical"])

with tab1:
    q = st.text_input("Ask about any scheduling concept:")
    if st.button("Ask", key="ask_btn") and q:
        with st.spinner("Thinking..."):
            response = answer_query(q)
            st.write(response)

with tab2:
    algo = st.selectbox("Algorithm", ["fcfs", "sjf", "srtf", "priority", "rr"])
    quantum = st.number_input("Quantum (RR only)", min_value=1, value=2) if algo == "rr" else None

    if "num_procs" not in st.session_state:
        st.session_state.num_procs = 3

    c1, c2 = st.columns(2)
    if c1.button("➕ Add process") and st.session_state.num_procs < 10:
        st.session_state.num_procs += 1
    if c2.button("➖ Remove process") and st.session_state.num_procs > 1:
        st.session_state.num_procs -= 1

    n = st.session_state.num_procs
    procs = []
    for i in range(n):
        col1, col2 = st.columns(2)
        a = col1.number_input(f"P{i+1} Arrival", min_value=0, value=i, key=f"a{i}")
        bt = col2.number_input(f"P{i+1} Burst", min_value=1, value=3, key=f"b{i}")
        procs.append(Process(f"P{i+1}", a, bt))

    if st.button("Solve", key="solve_btn"):
        fn = ALGORITHMS[algo]
        gantt, rows = fn(procs, quantum) if quantum else fn(procs)

        fig, ax = plt.subplots(figsize=(8, 1.5))
        for pid, s, e in gantt:
            ax.barh(0, e - s, left=s, edgecolor="black")
            ax.text((s + e) / 2, 0, pid, ha="center", va="center")
        ax.set_yticks([])
        st.pyplot(fig)
        st.table(rows)

        table_str = "\n".join(f"{p.pid}: Arrival={p.arrival}, Burst={p.burst}" for p in procs)
        instruction = (
            f"Solve the following process table using {algo.upper()} "
            f"scheduling and explain the steps."
        )
        
        with st.spinner("Asking the model to explain..."):
            llm_answer = answer_query(instruction, table_str, algo_key=algo, quantum=quantum)
        
        st.subheader("Model explanation")
        st.write(llm_answer)
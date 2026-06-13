# app.py — веб-приложение (Streamlit)
import streamlit as st
import pandas as pd
from core.engine import (
    COINS, predict_tomorrow, walk_forward,
    simulate, honest_check, START
)

st.set_page_config(page_title="Прогноз волатильности", layout="wide")

st.title("Предсказание волатильности крипты")
st.caption("Буря или штиль? Модель предсказывает СИЛУ движения, не направление.")

coin = st.selectbox("Выбери монету:", list(COINS.keys()))

if st.button("Анализировать", type="primary"):

    with st.spinner("Загружаю данные и считаю прогноз..."):
        pred = predict_tomorrow(coin)

    if pred is None:
        st.error("Мало данных для этой монеты")
        st.stop()

    st.subheader("Прогноз на завтра")
    p = pred["prob_storm"]
    if p > 0.55:
        st.success(f"БУРЯ — вероятность {p*100:.0f}%  (данные по {pred['date']})")
    elif p < 0.45:
        st.info(f"ШТИЛЬ — вероятность {(1-p)*100:.0f}%  (данные по {pred['date']})")
    else:
        st.warning(f"НЕЯСНО — {p*100:.0f}%  (данные по {pred['date']})")

    st.divider()

    st.subheader("Walk-forward проверка (честность модели)")
    with st.spinner("Прогоняю walk-forward..."):
        wf = walk_forward(coin)

    if wf:
        c1, c2 = st.columns(2)
        c1.metric("Средний AUC", f"{wf['mean']:.3f}", f"±{wf['std']:.3f}")
        verdict = "Сильный" if wf['mean'] > 0.55 else (
                  "Живой" if wf['mean'] > 0.52 else "Слабый")
        c2.metric("Вердикт", verdict)
        st.bar_chart(pd.DataFrame(
            {"AUC": wf["chunks"]},
            index=[f"Кусок {i+1}" for i in range(len(wf["chunks"]))]))

    st.divider()

    st.subheader("Симуляция $1000")
    with st.spinner("Считаю торговлю..."):
        sim = simulate(coin)

    if sim:
        c1, c2, c3 = st.columns(3)
        c1.metric("Стратегия 'буря'", f"${sim['strategy']:.0f}",
                  f"{(sim['strategy']/START-1)*100:+.0f}%")
        c2.metric("Buy & Hold", f"${sim['buyhold']:.0f}",
                  f"{(sim['buyhold']/START-1)*100:+.0f}%")
        c3.metric("Сделок", f"{sim['trades']} / {sim['days']}")

        if sim["strategy"] > sim["buyhold"]:
            st.success("Сигнал обыграл Buy & Hold!")
        else:
            st.info("Buy & Hold выиграл на этой монете")

    st.divider()

    st.subheader("Сигнал умнее случайности?")
    with st.spinner("Сравниваю с рандомом (100 попыток)..."):
        hc = honest_check(coin)

    if hc:
        c1, c2 = st.columns(2)
        c1.metric("Сигнал", f"${hc['signal']:.0f}")
        c2.metric("Рандом (среднее)", f"${hc['random']:.0f}")
        if hc["smart"]:
            st.success("Сигнал УМНЕЕ случайного входа! Реальная ценность есть.")
        else:
            st.warning("Сигнал примерно равен рандому.")

st.divider()
st.caption("Не финансовый совет. Прошлые результаты не гарантируют будущие.")
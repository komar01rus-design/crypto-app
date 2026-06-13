# app.py — веб-приложение (Streamlit)
import streamlit as st
import pandas as pd
import time
from core.engine import (
    COINS, predict_tomorrow, walk_forward,
    simulate, honest_check, START
)

st.set_page_config(page_title="Прогноз волатильности", layout="wide")

# ─── CSS терминал ───────────────────────────────────────────────
st.markdown("""
<style>
    .terminal {
        background-color: #000000;
        color: #00ff00;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #00ff00;
        min-height: 300px;
        max-height: 400px;
        overflow-y: auto;
        box-shadow: 0 0 15px rgba(0, 255, 0, 0.3);
    }
    .terminal p {
        margin: 3px 0;
        padding: 0;
    }
    .cursor {
        display: inline-block;
        width: 8px;
        height: 14px;
        background: #00ff00;
        animation: blink 1s infinite;
        vertical-align: middle;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }
    /* Блокируем взаимодействие во время загрузки */
    .block-ui {
        pointer-events: none;
        opacity: 0.5;
    }
</style>
""", unsafe_allow_html=True)


# ─── Функция терминала ──────────────────────────────────────────
def show_terminal(coin, lines, delay=0.35):
    """Показывает терминал с анимацией построчно"""
    placeholder = st.empty()
    displayed = []

    for line in lines:
        displayed.append(line)
        html = "<div class='terminal'>"
        for l in displayed:
            html += f"<p>{l}</p>"
        html += "<p><span class='cursor'></span></p>"
        html += "</div>"
        placeholder.markdown(html, unsafe_allow_html=True)
        time.sleep(delay)

    return placeholder


def terminal_lines(coin):
    return [
        f"<span style='color:#00ff00'>╔══════════════════════════════════════╗</span>",
        f"<span style='color:#00ff00'>║     CRYPTO VOLATILITY ANALYZER       ║</span>",
        f"<span style='color:#00ff00'>╚══════════════════════════════════════╝</span>",
        f"",
        f"<span style='color:#ffffff'>$</span> Инициализация системы...",
        f"<span style='color:#ffffff'>$</span> Подключение к Binance API... <span style='color:#00ff00'>✓ OK</span>",
        f"<span style='color:#ffffff'>$</span> Монета выбрана: <span style='color:#ffff00'>{coin}</span>",
        f"<span style='color:#ffffff'>$</span> Загрузка исторических данных...",
        f"<span style='color:#ffffff'>$</span> Обработка цен [<span style='color:#00ff00'>████████████████</span>] 100%",
        f"<span style='color:#ffffff'>$</span> Расчёт волатильности... <span style='color:#00ff00'>✓</span>",
        f"<span style='color:#ffffff'>$</span> RSI индикатор: <span style='color:#00ff00'>готов</span>",
        f"<span style='color:#ffffff'>$</span> MACD индикатор: <span style='color:#00ff00'>готов</span>",
        f"<span style='color:#ffffff'>$</span> Bollinger Bands: <span style='color:#00ff00'>готов</span>",
        f"<span style='color:#ffffff'>$</span> Запуск ML модели...",
        f"<span style='color:#ffffff'>$</span> Walk-forward валидация...",
        f"<span style='color:#ffffff'>$</span> Симуляция торговли $1000...",
        f"<span style='color:#ffffff'>$</span> Сравнение с рандомом (100 попыток)...",
        f"<span style='color:#ffffff'>$</span> Финальный анализ...",
        f"",
        f"<span style='color:#00ff00'>✓ АНАЛИЗ ЗАВЕРШЁН — загружаю результаты...</span>",
    ]


# ─── UI ─────────────────────────────────────────────────────────
st.title("Предсказание волатильности крипты")
st.caption("Буря или штиль? Модель предсказывает СИЛУ движения, не направление.")

coin = st.selectbox("Выбери монету:", list(COINS.keys()))

if st.button("Анализировать", type="primary"):

    # ── Показываем терминал и считаем всё параллельно ──
    terminal_placeholder = show_terminal(coin, terminal_lines(coin), delay=0.3)

    # Считаем все данные (терминал уже показан)
    pred = predict_tomorrow(coin)
    wf   = walk_forward(coin)
    sim  = simulate(coin)
    hc   = honest_check(coin)

    # Убираем терминал
    terminal_placeholder.empty()

    # ── Результаты ──────────────────────────────────────

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
    if wf:
        c1, c2 = st.columns(2)
        c1.metric("Средний AUC", f"{wf['mean']:.3f}", f"±{wf['std']:.3f}")
        verdict = "Сильный" if wf['mean'] > 0.55 else (
                  "Живой"   if wf['mean'] > 0.52 else "Слабый")
        c2.metric("Вердикт", verdict)
        st.bar_chart(pd.DataFrame(
            {"AUC": wf["chunks"]},
            index=[f"Кусок {i+1}" for i in range(len(wf["chunks"]))]))

    st.divider()

    st.subheader("Симуляция $1000")
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

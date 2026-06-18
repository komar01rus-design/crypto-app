# app.py — веб-приложение (Streamlit)
import streamlit as st
import pandas as pd
import time
from core.engine import (
    COINS, predict_tomorrow, walk_forward,
    simulate, honest_check, START
)

st.set_page_config(page_title="Прогноз волатильности", layout="wide")

# ─── CSS матричный стиль ────────────────────────────────────────
st.markdown("""
<style>
    /* Импорт шрифта */
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

    /* Фон и базовый текст */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #000000 !important;
        color: #00ff00 !important;
        font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    }

    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid #00ff00;
    }

    /* Заголовки */
    h1, h2, h3, h4, h5, h6 {
        color: #00ff00 !important;
        font-family: 'Share Tech Mono', monospace !important;
        text-shadow: 0 0 10px rgba(0, 255, 0, 0.5);
        letter-spacing: 2px;
    }

    /* Весь текст */
    p, span, div, label {
        color: #00cc00 !important;
        font-family: 'Share Tech Mono', monospace !important;
    }

    /* Caption */
    [data-testid="stCaptionContainer"] p {
        color: #007700 !important;
    }

    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {
        background-color: #000000 !important;
        border: 1px solid #00ff00 !important;
        color: #00ff00 !important;
        font-family: 'Share Tech Mono', monospace !important;
    }
    [data-testid="stSelectbox"] label {
        color: #00ff00 !important;
    }
    [data-baseweb="select"] {
        background-color: #000000 !important;
    }
    [data-baseweb="popover"] {
        background-color: #000000 !important;
        border: 1px solid #00ff00 !important;
    }
    li[role="option"] {
        background-color: #000000 !important;
        color: #00ff00 !important;
    }
    li[role="option"]:hover {
        background-color: #003300 !important;
    }

    /* Кнопка */
    [data-testid="stButton"] > button {
        background-color: #000000 !important;
        color: #00ff00 !important;
        border: 2px solid #00ff00 !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 16px !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
        box-shadow: 0 0 15px rgba(0, 255, 0, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stButton"] > button:hover {
        background-color: #003300 !important;
        box-shadow: 0 0 25px rgba(0, 255, 0, 0.8) !important;
        transform: scale(1.02);
    }

    /* Metric карточки */
    [data-testid="stMetric"] {
        background-color: #000000 !important;
        border: 1px solid #00ff00 !important;
        border-radius: 4px !important;
        padding: 15px !important;
        box-shadow: 0 0 10px rgba(0, 255, 0, 0.2) !important;
    }
    [data-testid="stMetricLabel"] p {
        color: #007700 !important;
        font-size: 12px !important;
        letter-spacing: 2px !important;
    }
    [data-testid="stMetricValue"] {
        color: #00ff00 !important;
        font-size: 28px !important;
        text-shadow: 0 0 8px rgba(0, 255, 0, 0.6) !important;
    }

    /* Divider */
    hr {
        border-color: #003300 !important;
    }

    /* Success / Info / Warning / Error */
    [data-testid="stSuccess"] {
        background-color: #001a00 !important;
        border: 1px solid #00ff00 !important;
        color: #00ff00 !important;
    }
    [data-testid="stInfo"] {
        background-color: #000d1a !important;
        border: 1px solid #0066ff !important;
        color: #0099ff !important;
    }
    [data-testid="stWarning"] {
        background-color: #1a1a00 !important;
        border: 1px solid #ffff00 !important;
        color: #ffff00 !important;
    }
    [data-testid="stError"] {
        background-color: #1a0000 !important;
        border: 1px solid #ff0000 !important;
        color: #ff0000 !important;
    }

    /* Spinner */
    [data-testid="stSpinner"] {
        color: #00ff00 !important;
    }

    /* ─── СКРЫВАЕМ GITHUB / МЕНЮ / ФУТЕР ─── */
    #MainMenu {
        display: none !important;
    }
    footer {
        display: none !important;
    }
    [data-testid="stToolbar"] {
        display: none !important;
    }
    [data-testid="stDeployButton"] {
        display: none !important;
    }
    [data-testid="stHeader"] {
        display: none !important;
    }
    /* Доп. селекторы на случай обновления Streamlit */
    .stActionButton {
        display: none !important;
    }
    button[title="View app in fullscreen"] {
        display: none !important;
    }
    button[title="Open GitHub repository"] {
        display: none !important;
    }
    /* ─────────────────────────────────────── */

    /* Bar chart */
    [data-testid="stArrowVegaLiteChart"] canvas {
        filter: hue-rotate(85deg) saturate(3) brightness(1.2);
    }

    /* Скроллбар */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #000000; }
    ::-webkit-scrollbar-thumb { background: #00ff00; border-radius: 3px; }

    /* Терминал */
    .terminal {
        background-color: #000000;
        color: #00ff00;
        font-family: 'Share Tech Mono', 'Courier New', monospace;
        font-size: 13px;
        padding: 20px;
        border-radius: 4px;
        border: 1px solid #00ff00;
        min-height: 320px;
        max-height: 420px;
        overflow-y: auto;
        box-shadow: 0 0 20px rgba(0, 255, 0, 0.35);
    }
    .terminal p {
        margin: 3px 0;
        padding: 0;
    }
    .cursor {
        display: inline-block;
        width: 8px;
        height: 13px;
        background: #00ff00;
        animation: blink 0.8s infinite;
        vertical-align: middle;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0; }
    }

    /* Заголовок */
    .spy-title {
        font-family: 'Share Tech Mono', monospace;
        color: #00ff00;
        font-size: 2.2em;
        text-shadow: 0 0 20px rgba(0,255,0,0.8), 0 0 40px rgba(0,255,0,0.4);
        letter-spacing: 4px;
        border-bottom: 1px solid #00ff00;
        padding-bottom: 10px;
        margin-bottom: 5px;
    }
    .spy-caption {
        color: #007700 !important;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 1px;
        font-size: 0.85em;
    }

</style>
""", unsafe_allow_html=True)


# ─── Функция терминала ──────────────────────────────────────────
def show_terminal(coin, lines, delay=0.3):
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
        "<span style='color:#005500'>╔══════════════════════════════════════════════╗</span>",
        "<span style='color:#005500'>║</span>   <span style='color:#00ff00'>CRYPTO INTELLIGENCE SYSTEM  v2.0</span>          <span style='color:#005500'>║</span>",
        "<span style='color:#005500'>║</span>   <span style='color:#007700'>CLASSIFIED </span>     <span style='color:#005500'>║</span>",
        "<span style='color:#005500'>╚══════════════════════════════════════════════╝</span>",
        "",
        f"<span style='color:#007700'>></span> Аутентификация агента... <span style='color:#00ff00'>ДОСТУП РАЗРЕШЁН</span>",
        f"<span style='color:#007700'>></span> Цель анализа: <span style='color:#00ff00'>{coin}</span>",
        f"<span style='color:#007700'>></span> Подключение к Binance API... <span style='color:#00ff00'>✓ ENCRYPTED</span>",
        f"<span style='color:#007700'>></span> Загрузка рыночных данных...",
        f"<span style='color:#007700'>></span> Дешифровка исторических цен [<span style='color:#00ff00'>████████████████</span>] 100%",
        f"<span style='color:#007700'>></span> Расчёт волатильности... <span style='color:#00ff00'>✓</span>",
        f"<span style='color:#007700'>></span> RSI:              <span style='color:#00ff00'>ПОЛУЧЕН</span>",
        f"<span style='color:#007700'>></span> MACD:             <span style='color:#00ff00'>ПОЛУЧЕН</span>",
        f"<span style='color:#007700'>></span> Bollinger Bands:  <span style='color:#00ff00'>ПОЛУЧЕН</span>",
        f"<span style='color:#007700'>></span> Запуск нейросети...",
        f"<span style='color:#007700'>></span> Walk-forward валидация...",
        f"<span style='color:#007700'>></span> Симуляция $1000...",
        f"<span style='color:#007700'>></span> Проверка против рандома (100 итераций)...",
        "",
        "<span style='color:#00ff00'>✓ АНАЛИЗ ЗАВЕРШЁН  — ВЫВОД РЕЗУЛЬТАТОВ...</span>",
    ]


# ─── Заголовок ──────────────────────────────────────────────────
st.markdown("<div class='spy-title'>⬛ CRYPTO VOLATILITY INTEL</div>", unsafe_allow_html=True)
st.markdown("<div class='spy-caption'>> БУРЯ ИЛИ ШТИЛЬ? // МОДЕЛЬ ПРЕДСКАЗЫВАЕТ СИЛУ ДВИЖЕНИЯ, НЕ НАПРАВЛЕНИЕ</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

coin = st.selectbox("ВЫБЕРИ ЦЕЛЬ:", list(COINS.keys()))

if st.button("⬛ ЗАПУСТИТЬ АНАЛИЗ", type="primary"):

    terminal_placeholder = show_terminal(coin, terminal_lines(coin), delay=0.28)

    pred = predict_tomorrow(coin)
    wf   = walk_forward(coin)
    sim  = simulate(coin)
    hc   = honest_check(coin)

    terminal_placeholder.empty()

    if pred is None:
        st.error("⛔ НЕДОСТАТОЧНО ДАННЫХ ДЛЯ АНАЛИЗА")
        st.stop()

    st.subheader("// ПРОГНОЗ НА ЗАВТРА")
    p = pred["prob_storm"]
    if p > 0.55:
        st.success(f"⚡ БУРЯ — вероятность {p*100:.0f}%  [ данные по {pred['date']} ]")
    elif p < 0.45:
        st.info(f"🌊 ШТИЛЬ — вероятность {(1-p)*100:.0f}%  [ данные по {pred['date']} ]")
    else:
        st.warning(f"⚠ НЕЯСНО — {p*100:.0f}%  [ данные по {pred['date']} ]")

    st.divider()

    st.subheader("// WALK-FORWARD ПРОВЕРКА")
    if wf:
        c1, c2 = st.columns(2)
        c1.metric("СРЕДНИЙ AUC", f"{wf['mean']:.3f}", f"±{wf['std']:.3f}")
        verdict = "СИЛЬНЫЙ" if wf['mean'] > 0.55 else (
                  "ЖИВОЙ"   if wf['mean'] > 0.52 else "СЛАБЫЙ")
        c2.metric("ВЕРДИКТ", verdict)
        st.bar_chart(
            pd.DataFrame(
                {"AUC": wf["chunks"]},
                index=[f"Кусок {i+1}" for i in range(len(wf["chunks"]))]
            ),
            color="#00ff00"
        )

    st.divider()

    st.subheader("// СИМУЛЯЦИЯ $1000")
    if sim:
        c1, c2, c3 = st.columns(3)
        c1.metric("СТРАТЕГИЯ 'БУРЯ'", f"${sim['strategy']:.0f}",
                  f"{(sim['strategy']/START-1)*100:+.0f}%")
        c2.metric("BUY & HOLD", f"${sim['buyhold']:.0f}",
                  f"{(sim['buyhold']/START-1)*100:+.0f}%")
        c3.metric("СДЕЛОК", f"{sim['trades']} / {sim['days']}")

        if sim["strategy"] > sim["buyhold"]:
            st.success("✓ СИГНАЛ ОБЫГРАЛ BUY & HOLD")
        else:
            st.info("ℹ BUY & HOLD ВЫИГРАЛ НА ЭТОЙ МОНЕТЕ")

    st.divider()

    st.subheader("// СИГНАЛ VS СЛУЧАЙНОСТЬ")
    if hc:
        c1, c2 = st.columns(2)
        c1.metric("СИГНАЛ", f"${hc['signal']:.0f}")
        c2.metric("РАНДОМ (среднее)", f"${hc['random']:.0f}")
        if hc["smart"]:
            st.success("✓ СИГНАЛ УМНЕЕ СЛУЧАЙНОГО ВХОДА — РЕАЛЬНАЯ ЦЕННОСТЬ ПОДТВЕРЖДЕНА")
        else:
            st.warning("⚠ СИГНАЛ ПРИМЕРНО РАВЕН РАНДОМУ")

st.divider()
st.markdown(
    "<div class='spy-caption'>> НЕ ЯВЛЯЕТСЯ ФИНАНСОВЫМ СОВЕТОМ // "
    "ПРОШЛЫЕ РЕЗУЛЬТАТЫ НЕ ГАРАНТИРУЮТ БУДУЩИЕ // CLASSIFIED</div>",
    unsafe_allow_html=True
)

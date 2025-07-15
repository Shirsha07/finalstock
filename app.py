import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
import ta

# App Title
st.image("logo_friendly.png",width=100)
st.title("Interactive Stock Market Dashboard")
st.sidebar.title("Options")

# Helper Functions
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    return stock.history(start=start_date, end=end_date)

def plot_candles_stick_bar(df, title="", currency="", show_ema=True, show_rsi=True, show_macd=True, show_atr=True):
    # Compute technical indicators
    df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd_diff()

    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
        subplot_titles=("Candlestick + EMA20" if show_ema else "Candlestick",
                        "Volume",
                        "RSI" if show_rsi else "",
                        "MACD" if show_macd else "",
                        "ATR" if show_atr else "")
    )

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name="Candlestick"
    ), row=1, col=1)

    # Add EMA20 if selected
    if show_ema:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA20'], mode="lines", name="EMA 20", line=dict(color='orange')
        ), row=1, col=1)

    # Volume chart
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], name="Volume"
    ), row=2, col=1)

    # Add RSI if selected
    if show_rsi:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'], mode="lines", name="RSI", line=dict(color='purple')
        ), row=3, col=1)

    # Add MACD if selected
    if show_macd:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD'], mode="lines", name="MACD", line=dict(color='green')
        ), row=4, col=1)

    # Add ATR if selected
    if show_atr:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['ATR'], mode="lines", name="ATR", line=dict(color='blue')
        ), row=5, col=1)

    fig.update_layout(
        title=f"{title} {currency}",
        template="plotly_dark",
        height=1000,
        showlegend=False,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig)

def plot_volume(data):
    fig = px.bar(data, x=data.index, y='Volume', title="Trading Volume", template="plotly_dark")
    st.plotly_chart(fig)

def plot_daily_returns(data):
    data['Daily Return'] = data['Close'].pct_change() * 100
    fig = px.line(data, x=data.index, y='Daily Return', title="Daily Returns (%)", template="plotly_dark")
    st.plotly_chart(fig)

def plot_moving_averages(data, windows):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name="Close Price"))
    for window in windows:
        data[f"MA{window}"] = data['Close'].rolling(window=window).mean()
        fig.add_trace(go.Scatter(x=data.index, y=data[f"MA{window}"], mode='lines', name=f"MA {window}"))
    fig.update_layout(title="Moving Averages", xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
    st.plotly_chart(fig)

def plot_correlation_matrix(data):
    corr = data.corr()
    fig = px.imshow(corr, title="Correlation Matrix", template="plotly_dark", text_auto=True, color_continuous_scale='RdBu_r')
    st.plotly_chart(fig)

st.html("""
  <style>
    [alt=Logo] {
      height: 3rem;
      width: auto;
      padding-left: 1rem;
    }
  </style>
""")

# Sidebar Inputs
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", value="RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", value=date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())

data = fetch_stock_data(ticker, start_date, end_date)

# Sidebar Checkboxes for Indicators
show_ema = st.sidebar.checkbox("EMA (20)", value=True)
show_rsi = st.sidebar.checkbox("RSI")
show_macd = st.sidebar.checkbox("MACD")
show_atr = st.sidebar.checkbox("ATR")

# Stock Visualizations
if not data.empty:
    st.subheader(f"Stock Data for {ticker}")
    st.write(data.tail())

    st.subheader("Candlestick + Indicators")
    plot_candles_stick_bar(data, title=ticker, show_ema=show_ema, show_rsi=show_rsi, show_macd=show_macd, show_atr=show_atr)

    st.subheader("Volume Chart")
    plot_volume(data)

    st.subheader("Daily Returns")
    plot_daily_returns(data)

    st.sidebar.header("Moving Averages")
    moving_averages = st.sidebar.multiselect("Select Moving Averages (days)", options=[10, 20, 50, 100, 200], default=[20, 50])
    if moving_averages:
        st.subheader("Moving Averages")
        plot_moving_averages(data, moving_averages)

#top gainers losers
st.markdown("---")
st.header("📊 Market Summary: Gainers, Losers & Trend")

# Define a list of sample tickers (replace with actual index tickers if desired)
market_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'INTC', 'CSCO']
price_changes = {}

for symbol in market_tickers:
    try:
        df = fetch_stock_data(symbol, start_date, end_date)
        if len(df) >= 2:
            change = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
            price_changes[symbol] = round(change, 2)
    except:
        continue

# Sort to get top gainers and losers
sorted_changes = dict(sorted(price_changes.items(), key=lambda item: item[1], reverse=True))
gainers = dict(list(sorted_changes.items())[:5])
losers = dict(list(sorted_changes.items())[-5:])

col1, col2 = st.columns(2)
with col1:
    st.subheader("🚀 Top Gainers")
    st.write(pd.DataFrame(gainers.items(), columns=["Ticker", "Change (%)"]))

with col2:
    st.subheader("📉 Top Losers")
    st.write(pd.DataFrame(losers.items(), columns=["Ticker", "Change (%)"]))

# PDF Chatbot
st.markdown("---")
st.subheader("🧠 Stock Market Chatbot")

with st.sidebar:
    st.header("Interact with Chatbot")
    pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

OPENAI_API_KEY = "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"  # Replace with your actual key securely

if pdf_file:
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()

    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generate embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User question input
    user_question = st.text_input("💬 Ask something about the document:")

    if user_question:
        with st.spinner("Thinking..."):
            similar_docs = vector_store.similarity_search(user_question)
            llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                temperature=0,
                model_name="gpt-3.5-turbo"
            )
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=similar_docs, question=user_question)

        st.subheader("📌 Answer")
        st.write(response)

# 📩 Contact Me
st.sidebar.markdown("---")
st.sidebar.markdown("📬 **Contact Me**")
if st.sidebar.button("Get in Touch"):
    st.sidebar.markdown("📧 Email: ismd07@gmail.com")
    st.sidebar.markdown("+91 6458792391")

st.logo("logo_yahoo_lightpurple.png", size="large")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Stock Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Live Stock Market Dashboard")
st.markdown("Real-time stock data, technical indicators "
            "and performance analysis.")
st.markdown("---")


st.sidebar.header("⚙️ Settings")

popular = {
    "Apple (AAPL)":      "AAPL",
    "Google (GOOGL)":    "GOOGL",
    "Microsoft (MSFT)":  "MSFT",
    "Tesla (TSLA)":      "TSLA",
    "Amazon (AMZN)":     "AMZN",
    "Meta (META)":       "META",
    "Nvidia (NVDA)":     "NVDA",
    "Reliance (RELIANCE.NS)": "RELIANCE.NS",
    "TCS (TCS.NS)":      "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS"
}

selected_name = st.sidebar.selectbox(
    "Select Stock:", list(popular.keys()))
ticker = popular[selected_name]

custom = st.sidebar.text_input(
    "Or enter custom ticker:", placeholder="e.g. NVDA, WIPRO.NS")
if custom.strip():
    ticker = custom.strip().upper()

period = st.sidebar.selectbox(
    "Time Period:",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=2
)

show_ma  = st.sidebar.checkbox("Show Moving Averages", value=True)
show_vol = st.sidebar.checkbox("Show Volume",          value=True)

# Load data
@st.cache_data(ttl=300)
def load_stock(ticker, period):
    stock = yf.Ticker(ticker)
    df    = stock.history(period=period)
    info  = stock.info
    return df, info

try:
    with st.spinner(f"Loading {ticker} data..."):
        df, info = load_stock(ticker, period)

    if df.empty:
        st.error(f"No data found for {ticker}. Check the ticker symbol.")
        st.stop()

    df.index = pd.to_datetime(df.index)

    
    current  = df['Close'].iloc[-1]
    prev     = df['Close'].iloc[-2]
    change   = current - prev
    change_p = (change / prev) * 100
    high_52w = df['Close'].max()
    low_52w  = df['Close'].min()
    avg_vol  = df['Volume'].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current Price",
                f"${current:.2f}",
                f"{change:+.2f} ({change_p:+.2f}%)")
    col2.metric("52W High",    f"${high_52w:.2f}")
    col3.metric("52W Low",     f"${low_52w:.2f}")
    col4.metric("Avg Volume",  f"{avg_vol/1e6:.1f}M")
    col5.metric("Data Points", f"{len(df)}")

    st.markdown("---")

   
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()

    
    if show_vol:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(13, 8),
            gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=(13, 6))

    ax1.plot(df.index, df['Close'],
             color='#3498db', linewidth=1.5, label='Close Price')

    if show_ma:
        ax1.plot(df.index, df['MA20'],
                 color='#e74c3c', linewidth=1.2,
                 linestyle='--', label='20-Day MA')
        ax1.plot(df.index, df['MA50'],
                 color='#2ecc71', linewidth=1.2,
                 linestyle='--', label='50-Day MA')

    ax1.fill_between(df.index, df['Close'],
                     df['Close'].min(), alpha=0.1, color='#3498db')
    ax1.set_title(f'{ticker} Stock Price', fontsize=14)
    ax1.set_ylabel('Price (USD)')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    if show_vol:
        colors = ['#2ecc71' if c >= o else '#e74c3c'
                  for c, o in zip(df['Close'], df['Open'])]
        ax2.bar(df.index, df['Volume'],
                color=colors, alpha=0.7, width=1)
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.grid(alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    
    left, right = st.columns(2)

    
    with left:
        st.markdown("### 📊 Monthly Returns")
        df['Return'] = df['Close'].pct_change()
        monthly = df['Return'].resample('ME').sum() * 100

        fig2, ax = plt.subplots(figsize=(7, 4))
        colors = ['#2ecc71' if x >= 0 else '#e74c3c'
                  for x in monthly.values]
        ax.bar(monthly.index, monthly.values,
               color=colors, edgecolor='black', width=20)
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_title('Monthly Returns (%)', fontsize=12)
        ax.set_ylabel('Return (%)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)

    
    with right:
        st.markdown("### 📋 Performance Stats")
        total_return = ((current - df['Close'].iloc[0]) /
                        df['Close'].iloc[0]) * 100
        volatility   = df['Return'].std() * np.sqrt(252) * 100
        sharpe       = (df['Return'].mean() /
                        df['Return'].std()) * np.sqrt(252)
        max_dd_val   = ((df['Close'] /
                         df['Close'].cummax()) - 1).min() * 100

        stats = pd.DataFrame({
            'Metric': [
                'Total Return',
                'Annualized Volatility',
                'Sharpe Ratio',
                'Max Drawdown',
                'Best Day',
                'Worst Day'
            ],
            'Value': [
                f"{total_return:+.2f}%",
                f"{volatility:.2f}%",
                f"{sharpe:.3f}",
                f"{max_dd_val:.2f}%",
                f"{df['Return'].max()*100:+.2f}%",
                f"{df['Return'].min()*100:+.2f}%"
            ]
        })
        st.dataframe(stats, use_container_width=True, hide_index=True)

      
        st.markdown("### 🕯️ Recent Data")
        recent = df[['Open', 'High', 'Low', 'Close',
                      'Volume']].tail(7).round(2)
        recent['Volume'] = (recent['Volume'] / 1e6).round(2)
        recent.index = recent.index.strftime('%b %d')
        recent.columns = ['Open', 'High', 'Low',
                           'Close', 'Vol(M)']
        st.dataframe(recent, use_container_width=True)

    st.markdown("---")

    
    st.markdown("### ⚖️ Compare Multiple Stocks")
    compare_input = st.text_input(
        "Enter tickers to compare (comma separated):",
        value="AAPL, GOOGL, MSFT, TSLA"
    )

    if st.button("Compare", type="primary"):
        tickers = [t.strip().upper()
                   for t in compare_input.split(',')]
        fig3, ax = plt.subplots(figsize=(13, 6))

        colors = ['#3498db', '#e74c3c',
                  '#2ecc71', '#f39c12',
                  '#9b59b6', '#1abc9c']

        for i, t in enumerate(tickers[:6]):
            try:
                data = yf.download(t, period=period,
                                   progress=False)
                if not data.empty:
                    close = data['Close'].squeeze()
                    norm  = (close / close.iloc[0]) * 100
                    ax.plot(norm.index, norm.values,
                            label=t,
                            color=colors[i % len(colors)],
                            linewidth=2)
            except:
                pass

        ax.set_title('Normalized Price Comparison (Base=100)',
                     fontsize=14)
        ax.set_ylabel('Normalized Price')
        ax.set_xlabel('Date')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter('%b %Y'))
        plt.tight_layout()
        st.pyplot(fig3)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Check your internet connection or try a "
            "different ticker symbol.")

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "Data: Yahoo Finance via yfinance | "
    "Refreshes every 5 minutes"
)
# Crypto Sentiment Forecasting

This project is an MSc dissertation focused on cryptocurrency price forecasting using a combination of market time-series data and sentiment-based indicators. The objective of the project is to examine whether sentiment information extracted from textual data can provide additional predictive value beyond traditional market-derived features for short-horizon crypto price movements.

The project is implemented as an end-to-end system. Market data and sentiment signals are processed and combined into a unified feature set, which is then used by a trained forecasting model to generate predictions. The outputs are stored as structured files and visualised through an interactive Streamlit dashboard that displays price history, sentiment indicators, and model forecasts.

To run the project locally, execute the following commands in order:

```bash
git clone https://github.com/gertidokaj/crypto-sentiment-forecasting.git
cd crypto-sentiment-forecasting

python -m venv .venv
source .venv/bin/activate    # macOS / Linux
# or
.venv\Scripts\activate       # Windows

pip install -r requirements.txt

streamlit run dashboard/app.py

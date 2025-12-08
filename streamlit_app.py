import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# === Настройка страницы ===
st.set_page_config(
    page_title="Монитор воздействия на окружающую среду",
    page_icon="🌍",
    layout="wide"
)


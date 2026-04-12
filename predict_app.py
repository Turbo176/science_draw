import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端，确保云端稳定运行
import pandas as pd

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="温度变化预测工具", layout="centered")
st.title("冷热水混合温度变化预测")
st.markdown("根据热平衡原理和指数衰减模型，预测不同初始温度的冷热水接触后的温度变化曲线。")

# ==========================================
# 侧边栏：用户输入
# ==========================================
st.sidebar.header("初始条件设置")

hot_init = st.sidebar.number_input("热水初始温度 (℃)", value=60.0, step=1.0, format="%.1f")
cold_init = st.sidebar.number_input("冷水初始温度 (℃)", value=20.0, step=1.0, format="%.1f")

# 假设质量相等，计算平衡温度
equilibrium_temp = (hot_init + cold_init) / 2
st.sidebar.markdown(f"预测平衡温度：{equilibrium_temp:.1f} ℃**")

# 时间常数 k (衰减速率) 可调节
k = st.sidebar.slider("温度变化速率 (k)", min_value=0.1, max_value=1.0, value=0.4, step=0.05,
                      help="值越大，温度变化越快（热交换越剧烈）。可根据实验数据调整。")

# 预测时长（分钟）
time_max = st.sidebar.slider("预测时长 (分钟)", min_value=5, max_value=30, value=10, step=1)

# ==========================================
# 核心预测函数
# ==========================================
def predict_temperatures(hot0, cold0, t_eq, k, t_max, num_points=100):
    t = np.linspace(0, t_max, num_points)
    hot_t = t_eq + (hot0 - t_eq) * np.exp(-k * t)
    cold_t = t_eq + (cold0 - t_eq) * np.exp(-k * t)
    return t, hot_t, cold_t

# ==========================================
# 生成预测曲线
# ==========================================
t, hot_pred, cold_pred = predict_temperatures(hot_init, cold_init, equilibrium_temp, k, time_max)

# ==========================================
# 绘制图表
# ==========================================
fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
ax.plot(t, hot_pred, 'r-', linewidth=2.5, label=f'Hot water ({hot_init}℃ → {equilibrium_temp:.1f}℃)')
ax.plot(t, cold_pred, 'b-', linewidth=2.5, label=f'Cold water ({cold_init}℃ → {equilibrium_temp:.1f}℃)')
ax.axhline(y=equilibrium_temp, color='gray', linestyle='--', linewidth=1, label=f'Equilibrium temp {equilibrium_temp:.1f}℃')

ax.set_xlabel('Time (min)', fontsize=12)
ax.set_ylabel('Temperature (℃)', fontsize=12)
ax.set_title('Temperature change prediction after mixing', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc='best')
ax.set_xlim(0, time_max)
ax.set_ylim(min(cold_init, equilibrium_temp) - 5, max(hot_init, equilibrium_temp) + 5)

# 直接展示图表（不提供下载按钮）
st.pyplot(fig)

# ==========================================
# 数据表格展示（可选）
# ==========================================
if st.checkbox("Show prediction data table"):
    # 选取部分时间点展示（例如每2分钟）
    show_times = np.arange(0, time_max + 0.1, 2)
    hot_vals = equilibrium_temp + (hot_init - equilibrium_temp) * np.exp(-k * show_times)
    cold_vals = equilibrium_temp + (cold_init - equilibrium_temp) * np.exp(-k * show_times)
    df = pd.DataFrame({
        "Time (min)": show_times,
        "Hot water (℃)": hot_vals,
        "Cold water (℃)": cold_vals
    })
    st.dataframe(df.style.format("{:.1f}"))

# ==========================================
# 简单原理说明
# ==========================================
with st.expander("Prediction model explanation"):
    st.markdown("""
    - **Heat balance principle**: Assuming the system is adiabatic and the masses of hot and cold water are equal, the final equilibrium temperature is `(T_hot + T_cold)/2`.
    - **Temperature change model**: Follows Newton's law of cooling, with temperature exponentially decaying over time:  
      \\[ T(t) = T_{\\text{eq}} + (T_0 - T_{\\text{eq}}) \\cdot e^{-k \\cdot t} \\]
    - **Parameter `k`**: Reflects the heat exchange rate. Larger values mean faster temperature change. Can be fitted from experimental data.
    - This tool provides qualitative predictions; actual curves may be affected by container material, heat loss to the environment, etc.
    """)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from scipy.integrate import odeint

# ==========================================
# 页面配置 (中文界面保持不变)
# ==========================================
st.set_page_config(page_title="温度变化预测工具", layout="centered")
st.title("冷热水温度变化预测")
st.markdown("模拟热水与冷水接触后，两者相互传热并与环境交换热量，最终都趋向室温。")

# ==========================================
# 侧边栏：用户输入 (中文保持不变)
# ==========================================
st.sidebar.header("初始条件设置")

hot_init = st.sidebar.number_input("热水初始温度 (℃)", value=60.0, step=1.0, format="%.1f")
cold_init = st.sidebar.number_input("冷水初始温度 (℃)", value=20.0, step=1.0, format="%.1f")
room_temp = st.sidebar.number_input("室温 (℃)", value=25.0, step=1.0, format="%.1f")

k_hc = st.sidebar.slider("热水-冷水热交换系数", min_value=0.0, max_value=1.0, value=0.4, step=0.05,
                         help="热水与冷水之间的传热速率，值越大热量交换越快")
k_h_env = st.sidebar.slider("热水-环境热交换系数", min_value=0.0, max_value=0.5, value=0.1, step=0.02,
                            help="热水向环境散热的速率")
k_c_env = st.sidebar.slider("冷水-环境热交换系数", min_value=0.0, max_value=0.5, value=0.1, step=0.02,
                            help="冷水向环境散热的速率")

time_max = st.sidebar.slider("预测时长 (分钟)", min_value=5, max_value=60, value=20, step=1)

# ==========================================
# 微分方程模型
# ==========================================
def heat_exchange(y, t, k_hc, k_h_env, k_c_env, room):
    T_h, T_c = y
    dT_h_dt = -k_hc * (T_h - T_c) - k_h_env * (T_h - room)
    dT_c_dt =  k_hc * (T_h - T_c) - k_c_env * (T_c - room)
    return [dT_h_dt, dT_c_dt]

# ==========================================
# 数值求解
# ==========================================
t = np.linspace(0, time_max, 200)
y0 = [hot_init, cold_init]
args = (k_hc, k_h_env, k_c_env, room_temp)
solution = odeint(heat_exchange, y0, t, args=args)
hot_vals, cold_vals = solution[:, 0], solution[:, 1]

# ==========================================
# 绘制图表 (图表文字全部改为英文)
# ==========================================
fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
ax.plot(t, hot_vals, 'r-', linewidth=2.5, 
        label=f'Hot water ({hot_init}°C → final ≈ {hot_vals[-1]:.1f}°C)')
ax.plot(t, cold_vals, 'b-', linewidth=2.5, 
        label=f'Cold water ({cold_init}°C → final ≈ {cold_vals[-1]:.1f}°C)')
ax.axhline(y=room_temp, color='gray', linestyle='--', linewidth=1, 
           label=f'Room temperature {room_temp:.1f}°C')

ax.set_xlabel('Time (min)', fontsize=12)
ax.set_ylabel('Temperature (°C)', fontsize=12)
ax.set_title('Prediction of Hot and Cold Water Temperature (Coupled Model)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc='best')
ax.set_xlim(0, time_max)
ax.set_ylim(min(cold_init, room_temp, hot_vals.min()) - 5, 
            max(hot_init, room_temp, hot_vals.max()) + 5)

st.pyplot(fig)

# ==========================================
# 数据表格展示 (可选，保持中文表头)
# ==========================================
if st.checkbox("显示预测数据表格"):
    show_times = np.arange(0, time_max + 0.1, 2)
    hot_interp = np.interp(show_times, t, hot_vals)
    cold_interp = np.interp(show_times, t, cold_vals)
    df = pd.DataFrame({
        "时间 (分钟)": show_times,
        "热水温度 (℃)": hot_interp,
        "冷水温度 (℃)": cold_interp
    })
    st.dataframe(df.style.format("{:.1f}"))

# ==========================================
# 原理说明 (保持中文)
# ==========================================
with st.expander("预测模型说明"):
    st.markdown(r"""
    **物理情景**：热水与冷水接触（例如试管中的冷水浸入热水烧杯），系统与外界环境（室温）存在热交换。  
    **数学模型**：基于牛顿冷却定律，热水温度 $T_h$ 和冷水温度 $T_c$ 满足：
    $$
    \begin{aligned}
    \frac{dT_h}{dt} &= -k_{hc}(T_h - T_c) - k_{h\text{env}}(T_h - T_{\text{room}}) \\
    \frac{dT_c}{dt} &=  k_{hc}(T_h - T_c) - k_{c\text{env}}(T_c - T_{\text{room}})
    \end{aligned}
    $$
    - $k_{hc}$：热水与冷水之间的热交换系数  
    - $k_{h\text{env}}$、$k_{c\text{env}}$：热水、冷水与环境的散热系数  
    - 最终两者都趋向室温 $T_{\text{room}}$，且相互影响。

    您可以通过侧边栏调节三个系数，观察不同传热速率对温度曲线的影响。
    """)

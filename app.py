import streamlit as st
import os
import json
import re
import base64
import numpy as np
import matplotlib
# 强制使用 Agg 后端，确保在无界面环境下稳定运行
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openai import OpenAI
import io

# ==========================================
# 0. 页面基础配置
# ==========================================
st.set_page_config(page_title="WD_Data_Extractor", layout="wide")

st.title("温度变化实验数据提取与可视化")
st.text("说明：本工具用于自动识别实验记录表数据并生成对比图。")

# ==========================================
# 1. 侧边栏配置
# ==========================================
st.sidebar.header("系统设置")
api_key = st.sidebar.text_input("请输入 DashScope API Key", type="password")
model_name = st.sidebar.selectbox("选择模型", ["qwen-vl-max", "qwen-vl-plus"])

if not api_key:
    st.sidebar.warning("请在左侧输入 API Key 以开始使用")

# ==========================================
# 2. 核心功能函数
# ==========================================
def extract_data_with_qwen(uploaded_file, client):
    """处理上传的图片文件，提取0~10分钟的温度数据"""
    base64_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    
    # 【修改点1】提示词要求提取0分钟到10分钟，共11个数据点
    prompt_text = """
    你是一个专业的数据分析助手。请查看这张图片中的实验记录表。
    任务：提取 0分钟（初始）、1分钟、2分钟、……、10分钟 的“热水温度”和“冷水温度”数据。
    注意：
    1. 忽略图片中可能出现的灰色回车符（弯曲箭头）。
    2. 请按常理识别（热水降温，冷水升温）。
    3. 数据顺序必须从0分钟开始到10分钟结束，一共11个数值。
    请严格按 JSON 格式返回，不要有任何多余文字：
    {
        "hot": [数值0, 数值1, 数值2, ..., 数值10],
        "cold": [数值0, 数值1, 数值2, ..., 数值10]
    }
    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            temperature=0.1
        )
        result_text = response.choices[0].message.content
        json_str = re.search(r'\{.*\}', result_text, re.DOTALL).group()
        return json.loads(json_str)
    except Exception as e:
        st.error(f"解析文件 {uploaded_file.name} 时出错: {e}")
        return None

# ==========================================
# 3. 页面主体：上传与处理
# ==========================================
uploaded_files = st.file_uploader("上传实验图片(可多选)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files and api_key:
    if st.button("开始批量处理并绘图"):
        client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        all_results = []
        progress_bar = st.progress(0)
        
        for index, file in enumerate(uploaded_files):
            res = extract_data_with_qwen(file, client)
            if res:
                all_results.append({"name": file.name, "data": res})
            progress_bar.progress((index + 1) / len(uploaded_files))
        
        if all_results:
            st.success(f"处理完成，共成功提取 {len(all_results)} 份数据")
            
            # 展示数据表格
            st.header("提取到的原始数据")
            for item in all_results:
                with st.expander(f"数据详情: {item['name']}"):
                    # 【修改点2】显示0~10分钟
                    st.write("热水温度 (0-10min):", item['data']['hot'])
                    st.write("冷水温度 (0-10min):", item['data']['cold'])

            # 综合绘图 - 全部使用英文标签，避免中文字体问题
            st.header("综合变化趋势图")
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            
            # 【修改点3】x轴改为0~10分钟
            x_time = np.arange(0, 11)
            hot_list, cold_list = [], []

            for item in all_results:
                h, c = item['data']['hot'], item['data']['cold']
                # 兼容性：如果模型返回11个数据则直接使用；如果只有10个（旧版），则补0分钟数据为NaN（不绘制）
                if len(h) == 11 and len(c) == 11:
                    hot_list.append(h)
                    cold_list.append(c)
                    ax.plot(x_time, h, color='red', alpha=0.1)
                    ax.plot(x_time, c, color='blue', alpha=0.1)
                elif len(h) == 10 and len(c) == 10:
                    # 对于旧格式（1-10分钟），显示警告但依然绘制在1-10分钟区间
                    st.warning(f"{item['name']} 缺少0分钟数据，将只绘制1~10分钟区间。建议重新提取。")
                    x_old = np.arange(1, 11)
                    hot_list.append(np.insert(h, 0, np.nan))   # 插入NaN，使长度变为11
                    cold_list.append(np.insert(c, 0, np.nan))
                    ax.plot(x_old, h, color='red', alpha=0.1, linestyle='--')
                    ax.plot(x_old, c, color='blue', alpha=0.1, linestyle='--')
                else:
                    st.warning(f"{item['name']} 数据长度异常 ({len(h)}, {len(c)})，跳过绘图")

            if hot_list:
                # 计算平均值时忽略NaN（即缺失的0分钟点）
                avg_hot = np.nanmean(hot_list, axis=0)
                avg_cold = np.nanmean(cold_list, axis=0)
                ax.plot(x_time, avg_hot, color='red', linewidth=3, label='Hot Water (Avg)')
                ax.plot(x_time, avg_cold, color='blue', linewidth=3, label='Cold Water (Avg)')
                
            ax.set_title("Experimental Data Summary (0-10 min)")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Temperature (°C)")
            ax.set_ylim(0, 110)
            ax.set_xticks(np.arange(0, 11, 1))   # 显示0到10的刻度
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 在网页端渲染图表
            st.pyplot(fig)
            
            # 提供下载
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                label="Download Chart",
                data=buf.getvalue(),
                file_name="summary_chart.png",
                mime="image/png"
            )

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
    """处理上传的图片文件"""
    base64_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    
    prompt_text = """
    你是一个专业的数据分析助手。请查看这张图片中的实验记录表。
    任务：提取 1到10分钟 的“热水温度”和“冷水温度”数据。
    注意：
    1. 忽略图片中可能出现的灰色回车符（弯曲箭头）。
    2. 请按常理识别（热水降温，冷水升温）。
    请严格按 JSON 格式返回，不要有任何多余文字：
    {
        "hot": [数值1, 数值2, ...一共10个],
        "cold": [数值1, 数值2, ...一共10个]
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
                    st.write("热水温度 (1-10min):", item['data']['hot'])
                    st.write("冷水温度 (1-10min):", item['data']['cold'])

            # 综合绘图
            st.header("综合变化趋势图")
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            
            # 尝试加载中文字体
            # plt.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK SC', 'sans-serif']
            # plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Serif CJK SC', 'AR PL UMing CN', 'WenQuanYi Zen Hei']
            plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
            
            x_time = np.arange(1, 11)
            hot_list, cold_list = [], []

            for item in all_results:
                h, c = item['data']['hot'], item['data']['cold']
                if len(h) == 10 and len(c) == 10:
                    hot_list.append(h)
                    cold_list.append(c)
                    ax.plot(x_time, h, color='red', alpha=0.1)
                    ax.plot(x_time, c, color='blue', alpha=0.1)

            if hot_list:
                ax.plot(x_time, np.mean(hot_list, axis=0), color='red', linewidth=3, label='热水(平均)')
                ax.plot(x_time, np.mean(cold_list, axis=0), color='blue', linewidth=3, label='冷水(平均)')
                
            ax.set_title("实验数据汇总分析")
            ax.set_xlabel("时间 (min)")
            ax.set_ylabel("温度 (deg C)")
            ax.set_ylim(0, 110)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 在网页端渲染图表
            st.pyplot(fig)
            
            # 提供下载
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                label="下载分析图表",
                data=buf.getvalue(),
                file_name="summary_chart.png",
                mime="image/png"
            )
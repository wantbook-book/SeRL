import gradio as gr
import json
import os
from typing import List, Dict, Any

def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件数据"""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"解析JSON行时出错: {e}")
                        continue
    return data

def format_response_display(response: str) -> str:
    """格式化响应文本以便更好地显示"""
    if not response:
        return "无响应内容"
    
    # 添加一些基本的格式化
    formatted = response.replace('\n\n', '\n\n---\n\n')
    return formatted

def get_problem_info(item: Dict[str, Any]) -> str:
    """获取问题信息"""
    info_parts = []
    
    if 'subject' in item:
        info_parts.append(f"**学科**: {item['subject']}")
    if 'level' in item:
        info_parts.append(f"**难度**: {item['level']}")
    if 'unique_id' in item:
        info_parts.append(f"**ID**: {item['unique_id']}")
    if 'answer' in item:
        info_parts.append(f"**标准答案**: {item['answer']}")
    
    return "\n".join(info_parts)

def compare_responses(data: List[Dict[str, Any]], index: int):
    """比较initial和revised响应"""
    if not data or index < 0 or index >= len(data):
        return "无数据", "无数据", "无数据", "无数据", "无数据"
    
    item = data[index]
    
    # 问题信息
    problem = item.get('problem', '无问题描述')
    problem_info = get_problem_info(item)
    
    # Initial响应
    initial_response = item.get('initial_response', '无初始响应')
    initial_formatted = format_response_display(initial_response)
    
    # Revised响应
    revised_response = item.get('revised_response', '无修订响应')
    revised_formatted = format_response_display(revised_response)
    
    # 批评信息
    critique_info = ""
    if 'critique_request' in item:
        critique_info += f"**批评请求**: {item['critique_request']}\n\n"
    if 'critique' in item:
        critique_info += f"**批评内容**: {item['critique']}\n\n"
    if 'edit_request' in item:
        critique_info += f"**修订请求**: {item['edit_request']}"
    
    return problem, problem_info, initial_formatted, revised_formatted, critique_info

def create_gradio_interface():
    """创建Gradio界面"""
    # 加载数据
    data_file = "/pubshare/fwk/code/SeRL/cai/output/sft_data.jsonl"
    data = load_jsonl_data(data_file)
    
    if not data:
        print(f"警告: 无法从 {data_file} 加载数据")
        data = []
    
    with gr.Blocks(title="Initial vs Revision 响应对比", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Initial vs Revision 响应对比工具")
        gr.Markdown(f"📊 当前加载了 **{len(data)}** 个样本")
        
        with gr.Row():
            with gr.Column(scale=1):
                index_slider = gr.Slider(
                    minimum=0, 
                    maximum=max(0, len(data) - 1), 
                    step=1, 
                    value=0, 
                    label="选择样本索引",
                    interactive=True
                )
                
                refresh_btn = gr.Button("🔄 刷新数据", variant="secondary")
        
        # 问题信息区域
        with gr.Group():
            gr.Markdown("## 📝 问题信息")
            problem_display = gr.Textbox(
                label="问题描述",
                lines=3,
                interactive=False
            )
            problem_info_display = gr.Markdown(label="问题详情")
        
        # 响应对比区域
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🎯 Initial 响应")
                initial_display = gr.Textbox(
                    label="初始响应",
                    lines=15,
                    interactive=False,
                    show_copy_button=True
                )
            
            with gr.Column():
                gr.Markdown("## ✨ Revised 响应")
                revised_display = gr.Textbox(
                    label="修订响应",
                    lines=15,
                    interactive=False,
                    show_copy_button=True
                )
        
        # 批评信息区域
        with gr.Group():
            gr.Markdown("## 💭 批评与修订信息")
            critique_display = gr.Markdown(label="批评详情")
        
        # 统计信息
        with gr.Group():
            gr.Markdown("## 📈 统计信息")
            stats_display = gr.Markdown(
                f"""- **总样本数**: {len(data)}
- **当前样本**: 1 / {len(data) if data else 0}
- **数据文件**: `{data_file}`"""
            )
        
        def update_display(index):
            """更新显示内容"""
            problem, problem_info, initial, revised, critique = compare_responses(data, int(index))
            
            stats = f"""- **总样本数**: {len(data)}
- **当前样本**: {int(index) + 1} / {len(data) if data else 0}
- **数据文件**: `{data_file}`"""
            
            return problem, problem_info, initial, revised, critique, stats
        
        def refresh_data():
            """刷新数据"""
            nonlocal data
            data = load_jsonl_data(data_file)
            
            # 更新slider的最大值
            new_max = max(0, len(data) - 1)
            
            # 返回更新后的组件
            return (
                gr.Slider(maximum=new_max, value=0),  # 更新slider
                f"📊 数据已刷新！当前加载了 **{len(data)}** 个样本",  # 更新标题
                *update_display(0)  # 更新显示内容
            )
        
        # 绑定事件
        index_slider.change(
            fn=update_display,
            inputs=[index_slider],
            outputs=[
                problem_display, 
                problem_info_display, 
                initial_display, 
                revised_display, 
                critique_display,
                stats_display
            ]
        )
        
        refresh_btn.click(
            fn=refresh_data,
            outputs=[
                index_slider,
                gr.Markdown(),  # 标题更新
                problem_display,
                problem_info_display,
                initial_display,
                revised_display,
                critique_display,
                stats_display
            ]
        )
        
        # 初始化显示
        if data:
            demo.load(
                fn=lambda: update_display(0),
                outputs=[
                    problem_display,
                    problem_info_display,
                    initial_display,
                    revised_display,
                    critique_display,
                    stats_display
                ]
            )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
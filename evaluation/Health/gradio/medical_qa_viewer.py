import gradio as gr
import json
import os
from pathlib import Path
import glob

class MedicalQAViewer:
    def __init__(self):
        self.data = []
        self.current_index = 0
        self.file_path = ""
        self.available_files = []
    
    def scan_directory(self, directory_path):
        """扫描目录中的JSONL文件"""
        if not directory_path or not os.path.exists(directory_path):
            return "目录不存在或路径为空", []
        
        if not os.path.isdir(directory_path):
            return "输入的路径不是目录", []
        
        try:
            # 递归搜索所有JSONL文件
            jsonl_files = glob.glob(os.path.join(directory_path, "**/*.jsonl"), recursive=True)
            
            if not jsonl_files:
                return "目录中没有找到JSONL文件", []
            
            # 按文件名排序
            jsonl_files.sort()
            self.available_files = jsonl_files
            
            # 返回相对路径用于显示
            relative_files = [os.path.relpath(f, directory_path) for f in jsonl_files]
            
            return f"找到 {len(jsonl_files)} 个JSONL文件", relative_files
        
        except Exception as e:
            return f"扫描目录时出错: {str(e)}", []
    
    def load_selected_file(self, selected_file_index):
        """加载选中的文件"""
        if not self.available_files or selected_file_index is None:
            return "没有可用文件或未选择文件", {}, {}, {}, {}, {}, {}, {}, {}, {}
        
        try:
            file_index = int(selected_file_index)
            if 0 <= file_index < len(self.available_files):
                selected_file = self.available_files[file_index]
                return self.load_jsonl_file(selected_file)
            else:
                return "选择的文件索引无效", {}, {}, {}, {}, {}, {}, {}, {}, {}
        except:
            return "文件索引格式错误", {}, {}, {}, {}, {}, {}, {}, {}, {}
    
    def load_jsonl_file(self, file_path):
        """加载JSONL文件"""
        if not file_path or not os.path.exists(file_path):
            return "文件不存在或路径为空", {}, {}, {}, {}, {}, {}, {}, {}, {}
        
        try:
            self.data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line.strip()))
            
            self.file_path = file_path
            self.current_index = 0
            
            if self.data:
                return self.display_current_item()
            else:
                return "文件为空", {}, {}, {}, {}, {}, {}, {}, {}, {}
        except Exception as e:
            return f"加载文件时出错: {str(e)}", {}, {}, {}, {}, {}, {}, {}, {}, {}
    
    def display_current_item(self):
        """显示当前项目的信息"""
        if not self.data:
            return "没有数据", {}, {}, {}, {}, {}, {}, {}, {}, {}
        
        item = self.data[self.current_index]
        
        # 基本信息
        basic_info = {
            "索引 (idx)": item.get("idx", "N/A"),
            "元信息 (meta_info)": item.get("meta_info", "N/A"),
            "正确答案索引 (answer_idx)": item.get("answer_idx", "N/A")
        }
        
        # 问题信息
        question_info = {
            "问题": item.get("question", "N/A")
        }
        
        # 选项信息
        options = item.get("options", {})
        options_info = {}
        for key, value in options.items():
            options_info[f"选项 {key}"] = value
        
        # 答案信息
        answer_info = {
            "正确答案": item.get("answer", "N/A")
        }
        
        # 生成的回答
        generated_responses = item.get("generated_response", [])
        if isinstance(generated_responses, list) and generated_responses:
            generated_text = generated_responses[0]
        else:
            generated_text = str(generated_responses) if generated_responses else "N/A"
        
        response_info = {
            "生成的回答": generated_text
        }
        
        # 评估结果信息
        evaluation_info = {
            "提取的答案 (extracted_answer)": item.get("extracted_answer", "N/A"),
            "是否正确 (is_correct)": item.get("is_correct", "N/A")
        }
        
        # 统计信息
        stats_info = {
            "回答长度 (response_length)": item.get("response_length", "N/A"),
            "Token数量 (token_count)": item.get("token_count", "N/A")
        }
        
        # 导航信息
        nav_info = f"当前: {self.current_index + 1} / {len(self.data)}"
        
        # 文件信息
        file_info = f"文件: {os.path.basename(self.file_path)}"
        
        return nav_info, basic_info, question_info, options_info, answer_info, response_info, evaluation_info, stats_info, file_info
    
    def next_item(self):
        """下一条"""
        if self.data and self.current_index < len(self.data) - 1:
            self.current_index += 1
        return self.display_current_item()
    
    def prev_item(self):
        """上一条"""
        if self.data and self.current_index > 0:
            self.current_index -= 1
        return self.display_current_item()
    
    def jump_to_index(self, index):
        """跳转到指定索引"""
        try:
            index = int(index) - 1  # 用户输入的是1-based，转换为0-based
            if 0 <= index < len(self.data):
                self.current_index = index
            return self.display_current_item()
        except:
            return self.display_current_item()

def create_interface():
    viewer = MedicalQAViewer()
    
    with gr.Blocks(title="医疗QA数据查看器", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 医疗QA数据查看器")
        gr.Markdown("上传JSONL文件或输入目录路径来查看医疗问答数据")
        
        # 文件加载方式选择
        with gr.Tab("直接上传文件"):
            with gr.Row():
                with gr.Column(scale=3):
                    file_input = gr.File(
                        label="选择JSONL文件",
                        file_types=[".jsonl"],
                        type="filepath"
                    )
                with gr.Column(scale=1):
                    load_btn = gr.Button("加载文件", variant="primary")
        
        with gr.Tab("从目录加载"):
            with gr.Row():
                with gr.Column(scale=3):
                    directory_input = gr.Textbox(
                        label="输入目录路径",
                        placeholder="例如: /pubshare/fwk/code/SeRL/evaluation/Health/outputs",
                        value="/pubshare/fwk/code/SeRL/evaluation/Health/outputs"
                    )
                with gr.Column(scale=1):
                    scan_btn = gr.Button("扫描目录", variant="secondary")
            
            scan_status = gr.Textbox(label="扫描状态", interactive=False)
            
            with gr.Row():
                with gr.Column(scale=3):
                    file_list = gr.Dropdown(
                        label="选择文件",
                        choices=[],
                        interactive=True
                    )
                with gr.Column(scale=1):
                    load_selected_btn = gr.Button("加载选中文件", variant="primary")
        
        # 导航区域
        with gr.Row():
            nav_info = gr.Textbox(label="导航信息", interactive=False)
            file_info = gr.Textbox(label="文件信息", interactive=False)
        
        with gr.Row():
            prev_btn = gr.Button("⬅️ 上一条", size="sm")
            jump_input = gr.Number(label="跳转到第几条", minimum=1, step=1, scale=1)
            jump_btn = gr.Button("跳转", size="sm")
            next_btn = gr.Button("下一条 ➡️", size="sm")
        
        # 数据显示区域
        with gr.Row():
            with gr.Column():
                basic_info = gr.JSON(label="基本信息", show_label=True)
                evaluation_info = gr.JSON(label="评估结果", show_label=True)
                stats_info = gr.JSON(label="统计信息", show_label=True)
            
            with gr.Column():
                question_info = gr.JSON(label="问题", show_label=True)
                options_info = gr.JSON(label="选项", show_label=True)
                answer_info = gr.JSON(label="正确答案", show_label=True)
        
        with gr.Row():
            response_info = gr.JSON(label="生成的回答", show_label=True)
        
        # 事件绑定
        load_btn.click(
            fn=viewer.load_jsonl_file,
            inputs=[file_input],
            outputs=[nav_info, basic_info, question_info, options_info, answer_info, response_info, evaluation_info, stats_info, file_info]
        )
        
        # 目录扫描事件
        scan_btn.click(
            fn=viewer.scan_directory,
            inputs=[directory_input],
            outputs=[scan_status, file_list]
        )
        
        # 加载选中文件事件
        load_selected_btn.click(
            fn=viewer.load_selected_file,
            inputs=[file_list],
            outputs=[nav_info, basic_info, question_info, options_info, answer_info, response_info, evaluation_info, stats_info, file_info]
        )
        
        next_btn.click(
            fn=viewer.next_item,
            outputs=[nav_info, basic_info, question_info, options_info, answer_info, response_info, evaluation_info, stats_info, file_info]
        )
        
        prev_btn.click(
            fn=viewer.prev_item,
            outputs=[nav_info, basic_info, question_info, options_info, answer_info, response_info, evaluation_info, stats_info, file_info]
        )
        
        jump_btn.click(
            fn=viewer.jump_to_index,
            inputs=[jump_input],
            outputs=[nav_info, basic_info, question_info, options_info, answer_info, response_info, evaluation_info, stats_info, file_info]
        )
        
        # 键盘快捷键
        demo.load(lambda: gr.update(value="使用 ⬅️ 和 ➡️ 按钮或跳转功能来浏览数据"), outputs=[nav_info])
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
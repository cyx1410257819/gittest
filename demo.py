import gradio as gr
import requests
import json
import tempfile
import soundfile as sf

class EmotionAwareChatSystem:
    def __init__(self):
        self.chat_history = []  # 聊天历史
        self.emotion_interrupted = False  # 情绪异常是否中断聊天
        # VLLM服务配置
        self.VLLM_BASE_URL = "http://100.64.0.26:8000/v1"
        self.CHAT_MODEL = "/root/cyx/model_weights/Qwen3-4B-Instruct-2507-FP8"  # 或 "Qwen2.5"
        self.EMOTION_MODEL = "/root/cyx/model_weights/Qwen3-4B-Instruct-2507-FP8"  # 用于情绪分析的模型

    def reset_chat(self):
        """重置聊天状态"""
        self.chat_history = []
        self.emotion_interrupted = False
        return [], "聊天已重置，可开始新对话"

    def call_vllm(self, messages, model, max_tokens=500, temperature=0.7):
        """调用VLLM API"""
        try:
            response = requests.post(
                f"{self.VLLM_BASE_URL}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"VLLM API调用失败: {response.status_code}, {response.text}")
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            raise Exception(f"VLLM API调用失败: {str(e)}")

    def check_emotion_with_vllm(self, text, history):
        """使用VLLM大模型分析文本情绪（仅基于当前用户问题）"""
        try:
            # 构建消息上下文（仅包含系统提示和当前用户文本）
            messages = [
                {"role": "system", "content": """您是一个情绪分析专家。请分析以下文本的情绪状态。
                识别可能的情绪异常，如：愤怒、焦虑、抑郁、极端负面情绪、攻击性语言等。
                如果检测到任何情绪异常，请返回：{"has_emotion_issue": true, "reason": "具体原因"}
                如果情绪正常，请返回：{"has_emotion_issue": false, "reason": "情绪正常"}"""},
                {"role": "user", "content": text}  # 仅添加当前用户的问题
            ]
            
            response_content = self.call_vllm(
                messages=messages,
                model=self.EMOTION_MODEL,
                max_tokens=512,
                temperature=0.7
            )
            
            # 解析VLLM返回的情绪分析结果
            try:
                result = json.loads(response_content)
                return result.get("has_emotion_issue", False), result.get("reason", "未知原因")
            except:
                # 如果不是JSON格式，尝试从文本中判断
                if "has_emotion_issue" in response_content.lower() or "情绪异常" in response_content:
                    return True, "大模型检测到情绪异常"
                return False, "情绪正常"
                
        except Exception as e:
            print(f"VLLM情绪分析出错: {e}")
            return False, "情绪分析服务暂时不可用"

    def check_emotion_with_asr(self, audio_file_path):
        """通过ASR服务获取语音的情绪标签"""
        try:
            with open(audio_file_path, "rb") as audio_file:
                files = {"file": audio_file}
                asr_response = requests.post("http://localhost:8024/asr", files=files)
            
            if asr_response.status_code != 200:
                return False, f"ASR服务调用失败: {asr_response.status_code}", ""
            
            asr_result = asr_response.json()
            if not asr_result.get("success"):
                return False, f"ASR识别失败: {asr_result.get('message', '未知错误')}", ""
            
            user_text = asr_result["clean_text"]
            emotion_tag = asr_result["emotion_tag"]
            is_abnormal = asr_result["is_abnormal"]
            
            reason = f"ASR检测到情绪标签: {emotion_tag}" if is_abnormal else "ASR情绪正常"
            
            return is_abnormal, reason, user_text
            
        except Exception as e:
            print(f"ASR情绪分析出错: {e}")
            return False, f"ASR情绪分析出错: {str(e)}", ""

    def process_audio(self, audio_data):
        """处理语音输入：ASR情绪检测 + VLLM内容分析 -> 决定是否聊天"""
        if self.emotion_interrupted:
            return self.chat_history, "⚠️ 聊天已因情绪异常中断，请重置对话"
        
        if audio_data is None:
            return self.chat_history, "❌ 未检测到音频输入"
        
        try:
            # 保存音频文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_data[1], samplerate=audio_data[0])
                audio_file_path = f.name
            
            # 1. 通过ASR获取文本和情绪标签
            asr_has_issue, asr_reason, user_text = self.check_emotion_with_asr(audio_file_path)
            
            if not user_text:
                return self.chat_history, f"❌ ASR识别失败: {asr_reason}"
            
            # 2. 使用VLLM分析文本内容情绪
            content_has_issue, content_reason = self.check_emotion_with_vllm(user_text, self.chat_history)
            
            # 3. 综合判断：只要任一模型检测到情绪异常就中断
            if asr_has_issue or content_has_issue:
                self.emotion_interrupted = True
                interrupt_reason = []
                if asr_has_issue:
                    interrupt_reason.append(f"ASR: {asr_reason}")
                if content_has_issue:
                    interrupt_reason.append(f"内容分析: {content_reason}")
                
                final_reason = "，".join(interrupt_reason)
                self.chat_history.append((user_text, f"⚠️ 检测到情绪异常：{final_reason}，聊天已中断"))
                return self.chat_history, "聊天已中断"
            
            # 4. 情绪正常，调用VLLM进行聊天
            vllm_response = self.call_vllm(
                messages=self.build_chat_messages(user_text),
                model=self.CHAT_MODEL,
                max_tokens=500,
                temperature=0.7
            )
            
            # 5. 更新聊天历史
            self.chat_history.append((user_text, vllm_response))
            return self.chat_history, "语音识别完成，聊天正常进行"
            
        except Exception as e:
            return self.chat_history, f"❌ 处理音频时出错: {str(e)}"

    def process_text(self, user_text):
        """处理文本输入：VLLM内容分析 -> 决定是否聊天"""
        if self.emotion_interrupted:
            return self.chat_history, "⚠️ 聊天已因情绪异常中断，请重置对话"
        
        if not user_text or not user_text.strip():
            return self.chat_history, "❌ 输入不能为空"
        
        try:
            # 1. 使用VLLM分析文本内容情绪
            content_has_issue, content_reason = self.check_emotion_with_vllm(user_text, self.chat_history)
            
            # 2. 检查上下文情绪波动变化
            context_has_issue, context_reason = self.check_emotion_context(user_text)
            
            # 3. 综合判断：任一检测到异常就中断
            if content_has_issue or context_has_issue:
                self.emotion_interrupted = True
                interrupt_reason = []
                if content_has_issue:
                    interrupt_reason.append(f"内容分析: {content_reason}")
                if context_has_issue:
                    interrupt_reason.append(f"上下文分析: {context_reason}")
                
                final_reason = "，".join(interrupt_reason)
                self.chat_history.append((user_text, f"⚠️ 检测到情绪异常：{final_reason}，聊天已中断"))
                return self.chat_history, "聊天已中断"
            
            # 4. 情绪正常，调用VLLM进行聊天
            vllm_response = self.call_vllm(
                messages=self.build_chat_messages(user_text),
                model=self.CHAT_MODEL,
                max_tokens=500,
                temperature=0.7
            )

            # 5. 更新聊天历史
            self.chat_history.append((user_text, vllm_response))
            return self.chat_history, "文本输入处理完成"
            
        except Exception as e:
            return self.chat_history, f"❌ 处理文本时出错: {str(e)}"
    
    def check_emotion_context(self, current_text):
        """通过上下文分析情绪波动变化"""
        if len(self.chat_history) < 2:
            return False, "上下文情绪正常"
        
        try:
            # 构建上下文分析消息
            messages = [
                {"role": "system", "content": """您是一个情绪波动分析专家。请分析对话的上下文，检测是否存在：
            1. 情绪急剧变化（如从平静突然变为愤怒）
            2. 持续的负面情绪积累
            3. 攻击性或威胁性语言的升级
            4. 极端情绪表达
            
            如果检测到任何情绪波动异常，请返回：{"has_emotion_issue": true, "reason": "具体原因"}
            否则返回：{"has_emotion_issue": false, "reason": "情绪稳定"}"""}
            ]
            
            # 添加最近的对话历史
            recent_history = self.chat_history[-5:]  # 最近5轮对话
            for user_msg, assistant_msg in recent_history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
            
            messages.append({"role": "user", "content": f"当前输入：{current_text}\n\n请分析这段对话的上下文情绪波动情况。"})
            
            response_content = self.call_vllm(
                messages=messages,
                model=self.EMOTION_MODEL,
                max_tokens=200,
                temperature=0.3
            )
            
            try:
                result = json.loads(response_content)
                return result.get("has_emotion_issue", False), result.get("reason", "上下文分析")
            except:
                return False, "上下文情绪正常"
                
        except Exception as e:
            print(f"上下文情绪分析出错: {e}")
            return False, "上下文分析服务暂时不可用"

    def build_chat_messages(self, user_text):
        """构建聊天消息格式"""
        messages = [{"role": "system", "content": "你是一个友好、专业的聊天助手。"}]
        for user_msg, assistant_msg in self.chat_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": user_text})
        
        return messages


# 构建Gradio界面
def create_interface():
    chat_system = EmotionAwareChatSystem()
    
    with gr.Blocks(title="情绪感知聊天系统") as demo:
        gr.Markdown("# 情绪感知聊天系统")
        gr.Markdown("集成SenseVoice ASR情绪识别 + 本地VLLM Qwen模型内容分析")
        gr.Markdown("语音或文本输入将经过双重情绪检测，任一模型检测到异常时自动中断聊天")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, label="聊天记录", type="messages")  # 修复：添加type参数
                status_msg = gr.Textbox(label="系统状态", interactive=False)
            
            with gr.Column(scale=1):
                audio_input = gr.Audio(sources=["microphone"], type="numpy", label="语音输入")  # 修复：sources参数
                text_input = gr.Textbox(lines=5, label="文本输入")
                with gr.Row():
                    audio_btn = gr.Button("发送语音", variant="primary")
                    text_btn = gr.Button("发送文本", variant="secondary")
                reset_btn = gr.Button("重置聊天", variant="stop")
        
        # 事件绑定
        audio_btn.click(
            chat_system.process_audio,
            inputs=[audio_input],
            outputs=[chatbot, status_msg]
        )
        text_btn.click(
            chat_system.process_text,
            inputs=[text_input],
            outputs=[chatbot, status_msg]
        )
        reset_btn.click(
            chat_system.reset_chat,
            outputs=[chatbot, status_msg]
        )
        
        # 添加回车键发送文本的功能
        text_input.submit(
            chat_system.process_text,
            inputs=[text_input],
            outputs=[chatbot, status_msg]
        )
        
        # 示例
        gr.Examples(
            examples=[
                ["我真的很生气，这太过分了！"],  # 预期中断
                ["你好，今天天气怎么样？"]       # 预期正常
            ],
            inputs=[text_input],
            outputs=[chatbot, status_msg],
            fn=chat_system.process_text
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
    server_name="0.0.0.0",
    server_port=7865,
    inbrowser=False,
    show_error=True,
    prevent_thread_lock=True
)

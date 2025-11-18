import fastapi
import uvicorn
import re
import tempfile
import soundfile as sf
from pydantic import BaseModel
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

app = fastapi.FastAPI(title="ASR情绪识别API")

# 初始化ASR模型
ASR_MODEL_PATH = "/root/cyx/model_weights/SenseVoiceSmall"  # 替换为实际路径
asr_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model=ASR_MODEL_PATH,
    model_revision="master",
    device="cuda:0"  # CPU用"cpu"
)

# 情绪标签解析配置
NORMAL_EMOTION = "NEUTRAL"
ABNORMAL_TAGS = {"ANGRY", "SAD", "ANXIOUS", "AGGRESSIVE"}

# 响应模型
class ASRResponse(BaseModel):
    clean_text: str
    emotion_tag: str
    is_abnormal: bool
    success: bool
    message: str

def parse_asr_result(raw_output):
    """解析ASR原始输出，提取文本和情绪标签"""
    try:
        raw_text = raw_output[0]['text']
        # 提取情绪标签（取第二个<|...|>标签）
        tags = re.findall(r'<\|(\w+)\|>', raw_text)
        emotion_tag = tags[1] if len(tags) >= 2 else "UNKNOWN"
        # 清洗文本
        clean_text = re.sub(r'<\|.*?\|>', '', raw_text).strip()
        # 判断是否异常
        is_abnormal = emotion_tag != NORMAL_EMOTION and emotion_tag in ABNORMAL_TAGS
        return {
            "clean_text": clean_text,
            "emotion_tag": emotion_tag,
            "is_abnormal": is_abnormal,
            "success": True,
            "message": "处理成功"
        }
    except Exception as e:
        return {
            "clean_text": "",
            "emotion_tag": "解析失败",
            "is_abnormal": False,
            "success": False,
            "message": str(e)
        }

@app.post("/asr", response_model=ASRResponse)
async def process_audio(file: fastapi.UploadFile = fastapi.File(...)):
    """处理上传的音频文件，返回文本和情绪分析结果"""
    try:
        # 保存上传的音频到临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            content = await file.read()
            f.write(content)
            # 调用ASR模型
            raw_result = asr_pipeline(f.name)
        # 解析结果
        return parse_asr_result(raw_result)
    except Exception as e:
        return {
            "clean_text": "",
            "emotion_tag": "处理失败",
            "is_abnormal": False,
            "success": False,
            "message": str(e)
        }

if __name__ == "__main__":
    # 启动ASR API服务（端口8000）
    uvicorn.run(app, host="0.0.0.0", port=8024)
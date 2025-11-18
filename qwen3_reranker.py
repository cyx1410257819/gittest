import logging
import math
import gc
import time
import torch
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from modelscope import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.inputs.data import TokensPrompt

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(title="Reranker API", description="Qwen3-Reranker 文档排序接口")

# 数据模型定义
class RerankRequest(BaseModel):
    task: str
    query: str  # 单个查询
    documents: List[str]  # 多个待排序文档
    topk: Optional[int] = None  # 可选，返回前topk个文档，默认返回全部

class RerankResponse(BaseModel):
    ranked_documents: List[Dict]  # 排序后的文档列表，包含文档内容和分数
    total_count: int  # 总文档数

# 模型和分词器初始化
class RerankerModel:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"
        self.suffix_tokens = None
        self.true_token = None
        self.false_token = None
        self.sampling_params = None
        self.max_length = 32768
        self.initialize()

    def initialize(self):
        """初始化模型和分词器"""
        try:
            logger.info("开始初始化模型和分词器...")
            number_of_gpu = torch.cuda.device_count()
            self.tokenizer = AutoTokenizer.from_pretrained('/root/cyx/model_weights/Qwen3-Reranker-4B')
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 初始化模型
            self.model = LLM(
                model='/root/cyx/model_weights/Qwen3-Reranker-4B',
                tensor_parallel_size=number_of_gpu,
                max_model_len=32768,
                enable_prefix_caching=True,
                gpu_memory_utilization=0.325
            )
            
            # 预处理必要的token
            self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
            self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
            self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
            
            # 配置采样参数
            self.sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                logprobs=20,
                allowed_token_ids=[self.true_token, self.false_token],
            )
            
            logger.info("模型和分词器初始化完成")
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise

    def format_instruction(self, instruction: str, query: str, doc: str) -> List[Dict]:
        """格式化指令"""
        return [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ]

    def process_inputs(self, query: str, documents: List[str], instruction: str) -> List[TokensPrompt]:
        """处理输入数据，生成查询-文档对的token"""
        # 为每个文档创建与查询的配对
        messages = [self.format_instruction(instruction, query, doc) for doc in documents]
        messages = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=False, 
            enable_thinking=False
        )
        # 确保不超过最大长度
        max_content_length = self.max_length - len(self.suffix_tokens)
        messages = [ele[:max_content_length] + self.suffix_tokens for ele in messages]
        return [TokensPrompt(prompt_token_ids=ele) for ele in messages]

    def rank_documents(self, query: str, documents: List[str], instruction: str, topk: Optional[int] = None) -> List[Dict]:
        """对文档进行排序并返回前topk个"""
        try:
            if not documents:
                return []
                
            # 处理输入并获取分数
            inputs = self.process_inputs(query, documents, instruction)
            outputs = self.model.generate(inputs, self.sampling_params, use_tqdm=False)
            
            # 计算每个文档的分数
            doc_scores = []
            for i, output in enumerate(outputs):
                final_logits = output.outputs[0].logprobs[-1]
                
                # 获取"yes"和"no"的log概率
                true_logit = final_logits[self.true_token].logprob if self.true_token in final_logits else -10
                false_logit = final_logits[self.false_token].logprob if self.false_token in final_logits else -10
                
                # 计算概率比例作为分数
                true_score = math.exp(true_logit)
                false_score = math.exp(false_logit)
                score = true_score / (true_score + false_score)
                doc_scores.append({
                    "document": documents[i],
                    "score": round(score, 6)
                })
            
            # 按分数降序排序
            doc_scores.sort(key=lambda x: x["score"], reverse=True)
            
            # 截取前topk个
            if topk and topk > 0:
                doc_scores = doc_scores[:topk]
                
            return doc_scores
        except Exception as e:
            logger.error(f"文档排序时出错: {str(e)}")
            raise

# 全局模型实例
reranker = RerankerModel()

# API端点
@app.post("/rank_documents", response_model=RerankResponse, description="对查询相关的文档进行排序并返回前topk个")
async def rank_documents(request: RerankRequest):
    try:
        # 验证输入
        if not request.documents:
            raise HTTPException(status_code=400, detail="documents列表不能为空")
        
        if request.topk is not None and (request.topk <= 0 or request.topk > len(request.documents)):
            raise HTTPException(
                status_code=400, 
                detail=f"topk必须为正数且不大于文档总数({len(request.documents)})"
            )
        
        # 排序文档
        ranked_docs = reranker.rank_documents(
            query=request.query,
            documents=request.documents,
            instruction=request.task,
            topk=request.topk
        )
        
        return {
            "ranked_documents": ranked_docs,
            "total_count": len(request.documents)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API请求处理出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", description="健康检查接口")
async def health_check():
    return {"status": "healthy", "model": "Qwen3-Reranker-0.6B"}

# 程序退出时清理资源
@app.on_event("shutdown")
def shutdown_event():
    logger.info("正在关闭模型...")
    time.sleep(1)  # 增加延迟确保资源释放
    destroy_model_parallel()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("服务已关闭")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
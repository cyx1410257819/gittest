import logging
import math
import gc
import time
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM
from vllm.distributed.parallel_state import destroy_model_parallel

# -------------------------- 日志配置 --------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------- FastAPI 初始化 --------------------------
app = FastAPI(
    title="Qwen3-Embedding-0.6B Reranker API",
    description="基于 Qwen3-Embedding-0.6B 的相似度文档排序接口",
)

# -------------------------- 请求与响应模型 --------------------------
class RerankRequest(BaseModel):
    task: str
    query: str
    documents: List[str]
    topk: Optional[int] = 5
    normalize: Optional[bool] = True

class RerankResponse(BaseModel):
    ranked_documents: List[Dict]
    total_count: int

# -------------------------- 模型封装 --------------------------
class EmbeddingReranker:
    def __init__(self):
        self.model = None
        self.initialize()

    def initialize(self):
        """初始化模型"""
        try:
            logger.info("🚀 开始加载 Qwen3-Embedding-0.6B 模型...")
            num_gpus = torch.cuda.device_count()
            self.model = LLM(
                model="/root/cyx/model_weights/Qwen3-Embedding-0.6B",
                task="embed",
                tensor_parallel_size=num_gpus if num_gpus > 0 else 1,
                gpu_memory_utilization=0.15,
            )
            logger.info("✅ 模型加载完成")
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise

    def embed(self, task: str, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """生成embedding并返回tensor"""
        try:
            inputs = [f"Instruct: {task}\nQuery: {t}" for t in texts]
            outputs = self.model.embed(inputs)
            embeddings = [o.outputs.embedding for o in outputs]
            emb_tensor = torch.tensor(embeddings)
            if normalize:
                emb_tensor = F.normalize(emb_tensor, p=2, dim=1)
            return emb_tensor
        except Exception as e:
            logger.error(f"生成embedding失败: {str(e)}")
            raise

    def rank_documents(
        self,
        task: str,
        query: str,
        documents: List[str],
        topk: Optional[int] = 5,
        normalize: bool = True,
    ) -> List[Dict]:
        """计算相似度并返回前topk文档"""
        if not documents:
            return []

        # 生成query + documents的向量
        inputs = [f"Instruct: {task}\nQuery: {query}"] + documents
        embeddings = self.embed(task, inputs, normalize)
        query_emb = embeddings[0].unsqueeze(0)
        doc_embs = embeddings[1:]

        # 计算余弦相似度
        scores = (query_emb @ doc_embs.T).squeeze(0)
        scores = scores.tolist()

        # 排序
        ranked = sorted(
            [{"document": doc, "score": round(float(score), 6)} for doc, score in zip(documents, scores)],
            key=lambda x: x["score"],
            reverse=True,
        )

        # 返回前topk
        if topk and topk > 0:
            ranked = ranked[:topk]
        return ranked


# -------------------------- 全局模型实例 --------------------------
reranker = EmbeddingReranker()

# -------------------------- API 路由 --------------------------
@app.post("/rank_documents", response_model=RerankResponse)
async def rank_documents(request: RerankRequest):
    try:
        if not request.documents:
            raise HTTPException(status_code=400, detail="documents不能为空")
        if request.topk and (request.topk <= 0 or request.topk > len(request.documents)):
            raise HTTPException(status_code=400, detail=f"topk必须在1到{len(request.documents)}之间")

        ranked = reranker.rank_documents(
            task=request.task,
            query=request.query,
            documents=request.documents,
            topk=request.topk,
            normalize=request.normalize,
        )

        return {"ranked_documents": ranked, "total_count": len(request.documents)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"请求处理出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "Qwen3-Embedding-0.6B"}

@app.on_event("shutdown")
def shutdown_event():
    logger.info("正在释放模型资源...")
    time.sleep(1)
    destroy_model_parallel()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("服务已关闭 ✅")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, workers=1)


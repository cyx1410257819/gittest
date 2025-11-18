import logging
import gc
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from modelscope import AutoModel
from vllm import LLM
from vllm.distributed.parallel_state import destroy_model_parallel

# --------------------- 日志配置 ---------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------- FastAPI 初始化 ---------------------
app = FastAPI(
    title="Hybrid Reranker API",
    description="基于 KaLM-Embedding-2.5 与 Jina-Reranker-v3 的两阶段文档排序接口",
)

# --------------------- 请求与响应模型 ---------------------
class HybridRerankRequest(BaseModel):
    task: str
    query: str
    documents: List[str]
    topk: Optional[int] = 5
    threshold: Optional[float] = 0.6
    normalize: Optional[bool] = True

class HybridRerankResponse(BaseModel):
    ranked_documents: List[Dict]
    total_count: int

# ==========================================================
#                   模型1：Embedding 粗排
# ==========================================================
class EmbeddingReranker:
    def __init__(self):
        self.model = None
        self.initialize()

    def initialize(self):
        try:
            logger.info("🚀 加载 KaLM-Embedding-2.5 模型...")
            num_gpus = torch.cuda.device_count()
            self.model = LLM(
                model="/root/cyx/model_weights/KaLM-Embedding-2.5",  # KaLM模型路径
                task="embed",
                tensor_parallel_size=num_gpus if num_gpus > 0 else 1,
                gpu_memory_utilization=0.15,
                trust_remote_code=True,
                dtype="float16",
            )
            logger.info("✅ KaLM-Embedding 模型加载完成")
        except Exception as e:
            logger.error(f"❌ KaLM-Embedding 初始化失败: {e}")
            raise

    def embed(self, task: str, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """生成文本embedding"""
        try:
            inputs = [f"Instruct: {task}\nQuery: {t}" for t in texts]
            outputs = self.model.embed(inputs)
            embeddings = [o.outputs.embedding for o in outputs]
            emb_tensor = torch.tensor(embeddings)
            if normalize:
                emb_tensor = F.normalize(emb_tensor, p=2, dim=1)
            return emb_tensor
        except Exception as e:
            logger.error(f"生成embedding失败: {e}")
            raise

    def rank_documents(
        self, task: str, query: str, documents: List[str], normalize: bool = True
    ) -> List[Dict]:
        """根据embedding相似度进行粗排"""
        if not documents:
            return []
        inputs = [f"Instruct: {task}\nQuery: {query}"] + documents
        embeddings = self.embed(task, inputs, normalize)
        query_emb = embeddings[0].unsqueeze(0)
        doc_embs = embeddings[1:]
        scores = (query_emb @ doc_embs.T).squeeze(0).tolist()

        ranked = sorted(
            [{"document": doc, "score": float(score)} for doc, score in zip(documents, scores)],
            key=lambda x: x["score"],
            reverse=True,
        )
        return ranked

# ==========================================================
#                   模型2：Reranker 精排 (Jina-Reranker-v3)
# ==========================================================
class JinaReranker:
    def __init__(self):
        self.model = None
        self.initialize()

    def initialize(self):
        try:
            logger.info("🚀 加载 Jina-Reranker-v3 模型...")
            # 本地模型路径（添加local://前缀避免格式校验错误）
            model_path = "/root/cyx/model_weights/jina-reranker-v3"
            
            self.model = AutoModel.from_pretrained(
                model_path,
                dtype="auto",
                trust_remote_code=True,
            )
            self.model.eval()  # 评估模式
            
            # 移至GPU（如果可用）
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            
            logger.info("✅ Jina-Reranker-v3 模型加载完成")
        except Exception as e:
            logger.error(f"❌ Jina-Reranker-v3 初始化失败: {e}")
            raise

    def rank_documents(
        self, query: str, documents: List[str]  # 仅使用query，不依赖task
    ) -> List[Dict]:
        """基于Jina-Reranker进行精排打分"""
        if not documents:
            return []
        
        # 关闭梯度计算，节省内存
        with torch.no_grad():
            results = self.model.rerank(query, documents)
        
        # 转换结果格式（确保分数为Python float类型）
        ranked_results = [
            {
                "document": result["document"],
                "score": round(float(result["relevance_score"]), 6)  # 转换numpy类型为Python float
            } 
            for result in results
        ]
        
        # 按分数降序排序
        return sorted(ranked_results, key=lambda x: x["score"], reverse=True)

# ==========================================================
#                   混合排序控制逻辑
# ==========================================================
class HybridReranker:
    def __init__(self):
        self.embed_model = EmbeddingReranker()
        self.reranker = JinaReranker()

    def hybrid_rank(
        self, task: str, query: str, documents: List[str], topk: int, threshold: float, normalize: bool
    ) -> List[Dict]:
        # Step 1: Embedding 粗排（使用task优化粗排效果）
        logger.info("🔹 阶段1: Embedding 粗排中...")
        embedding_rank = self.embed_model.rank_documents(task, query, documents, normalize)
        # 取粗排前3*topk作为精排候选（平衡效率与召回）
        candidates = embedding_rank[: min(len(embedding_rank), 3 * topk)]
        candidate_docs = [d["document"] for d in candidates]

        # Step 2: Reranker 精排（仅用query，符合Jina模型特性）
        logger.info("🔹 阶段2: Reranker 精排中...")
        reranked = self.reranker.rank_documents(query, candidate_docs)

        # Step 3: 阈值过滤
        reranked_filtered = [d for d in reranked if d["score"] >= threshold]

        # Step 4: 不足topk时从粗排结果回补
        if len(reranked_filtered) < topk:
            existing_docs = {d["document"] for d in reranked_filtered}
            for d in embedding_rank:
                if len(reranked_filtered) >= topk:
                    break
                if d["document"] not in existing_docs:
                    reranked_filtered.append(d)
                    existing_docs.add(d["document"])

        # 最终排序并截取topk
        reranked_filtered.sort(key=lambda x: x["score"], reverse=True)
        return reranked_filtered[:topk]

# ==========================================================
#                   全局实例 & API
# ==========================================================
hybrid_reranker = HybridReranker()


@app.post("/hybrid_rank", response_model=HybridRerankResponse)
async def hybrid_rank(request: HybridRerankRequest):
    try:
        if not request.documents:
            raise HTTPException(status_code=400, detail="documents不能为空")
        
        # 处理topk边界情况
        effective_topk = min(request.topk, len(request.documents))
        if request.topk <= 0:
            raise HTTPException(status_code=400, detail="topk必须大于0")

        results = hybrid_reranker.hybrid_rank(
            task=request.task,
            query=request.query,
            documents=request.documents,
            topk=effective_topk,
            threshold=request.threshold,
            normalize=request.normalize,
        )
        return {"ranked_documents": results, "total_count": len(request.documents)}

    except Exception as e:
        logger.error(f"请求处理出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models": ["KaLM-Embedding-2.5", "Jina-Reranker-v3"]}

@app.on_event("shutdown")
def shutdown_event():
    logger.info("🧹 正在释放模型资源...")
    destroy_model_parallel()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("✅ 服务已安全关闭")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8074, workers=1)
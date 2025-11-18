import logging
import math
import gc
import time
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from modelscope import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt
from vllm.distributed.parallel_state import destroy_model_parallel

# --------------------- 日志配置 ---------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------- FastAPI 初始化 ---------------------
app = FastAPI(
    title="Qwen3 Hybrid Reranker API",
    description="基于 Qwen3-Embedding 与 Qwen3-Reranker 的两阶段文档排序接口",
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
            logger.info("🚀 加载 Qwen3-Embedding-0.6B 模型...")
            num_gpus = torch.cuda.device_count()
            self.model = LLM(
                model="/root/cyx/model_weights/Qwen3-Embedding-0.6B",
                task="embed",
                tensor_parallel_size=num_gpus if num_gpus > 0 else 1,
                gpu_memory_utilization=0.15,
            )
            logger.info("✅ Qwen3-Embedding 模型加载完成")
        except Exception as e:
            logger.error(f"❌ Qwen3-Embedding 初始化失败: {e}")
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
#                   模型2：Reranker 精排
# ==========================================================
class QwenReranker:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"
        self.suffix_tokens = None
        self.true_token = None
        self.false_token = None
        self.sampling_params = None
        self.max_length = 32768
        self.initialize()

    def initialize(self):
        try:
            logger.info("🚀 加载 Qwen3-Reranker-4B 模型...")
            num_gpus = torch.cuda.device_count()
            self.tokenizer = AutoTokenizer.from_pretrained("/root/cyx/model_weights/Qwen3-Reranker-4B")
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = LLM(
                model="/root/cyx/model_weights/Qwen3-Reranker-4B",
                tensor_parallel_size=num_gpus,
                max_model_len=self.max_length,
                enable_prefix_caching=True,
                gpu_memory_utilization=0.325,
            )

            self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
            self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
            self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]

            self.sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                logprobs=20,
                allowed_token_ids=[self.true_token, self.false_token],
            )
            logger.info("✅ Qwen3-Reranker 模型加载完成")
        except Exception as e:
            logger.error(f"❌ Qwen3-Reranker 初始化失败: {e}")
            raise

    def format_instruction(self, instruction: str, query: str, doc: str) -> List[Dict]:
        return [
            {
                "role": "system",
                "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. The answer must be 'yes' or 'no'.",
            },
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"},
        ]

    def process_inputs(self, instruction: str, query: str, documents: List[str]) -> List[TokensPrompt]:
        messages = [self.format_instruction(instruction, query, doc) for doc in documents]
        messages = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        max_len = self.max_length - len(self.suffix_tokens)
        messages = [ele[:max_len] + self.suffix_tokens for ele in messages]
        return [TokensPrompt(prompt_token_ids=ele) for ele in messages]

    def rank_documents(
        self, instruction: str, query: str, documents: List[str]
    ) -> List[Dict]:
        """基于 reranker 进行打分"""
        if not documents:
            return []
        inputs = self.process_inputs(instruction, query, documents)
        outputs = self.model.generate(inputs, self.sampling_params, use_tqdm=False)
        scores = []
        for i, output in enumerate(outputs):
            final_logits = output.outputs[0].logprobs[-1]
            true_logit = final_logits[self.true_token].logprob if self.true_token in final_logits else -10
            false_logit = final_logits[self.false_token].logprob if self.false_token in final_logits else -10
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            scores.append({"document": documents[i], "score": round(score, 6)})
        return sorted(scores, key=lambda x: x["score"], reverse=True)

# ==========================================================
#                   混合排序控制逻辑
# ==========================================================
class HybridReranker:
    def __init__(self):
        self.embed_model = EmbeddingReranker()
        self.reranker = QwenReranker()

    def hybrid_rank(
        self, task: str, query: str, documents: List[str], topk: int, threshold: float, normalize: bool
    ) -> List[Dict]:
        # Step 1: Embedding 粗排
        logger.info("🔹 阶段1: Embedding 粗排中...")
        embedding_rank = self.embed_model.rank_documents(task, query, documents, normalize)
        candidates = embedding_rank[: min(len(embedding_rank), 3 * topk)]
        candidate_docs = [d["document"] for d in candidates]

        # Step 2: Reranker 精排
        logger.info("🔹 阶段2: Reranker 精排中...")
        reranked = self.reranker.rank_documents(task, query, candidate_docs)

        # Step 3: 阈值过滤
        reranked_filtered = [d for d in reranked if d["score"] >= threshold]

        # Step 4: 不足topk时回补
        if len(reranked_filtered) < topk:
            existing_docs = {d["document"] for d in reranked_filtered}
            for d in embedding_rank:
                if len(reranked_filtered) >= topk:
                    break
                if d["document"] not in existing_docs:
                    reranked_filtered.append(d)
                    existing_docs.add(d["document"])

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
        if request.topk <= 0 or request.topk > len(request.documents):
            raise HTTPException(status_code=400, detail=f"topk必须在1到{len(request.documents)}之间")

        results = hybrid_reranker.hybrid_rank(
            task=request.task,
            query=request.query,
            documents=request.documents,
            topk=request.topk,
            threshold=request.threshold,
            normalize=request.normalize,
        )
        return {"ranked_documents": results, "total_count": len(request.documents)}

    except Exception as e:
        logger.error(f"请求处理出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models": ["Qwen3-Embedding-0.6B", "Qwen3-Reranker-4B"]}

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
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)

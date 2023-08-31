"""
===========================================
        Module: Open-source LLM Setup
===========================================
"""
from langchain.llms import CTransformers

def build_llm(cfg):
    # Local CTransformers model
    if cfg.USE_GPU:
        llm = CTransformers(
            model=cfg.MODEL_BIN_PATH,
            model_type=cfg.MODEL_TYPE,
            device="cuda",
            batch_size=cfg.MODEL_BATCH_SIZE,
            config={
                "max_new_tokens": cfg.MAX_NEW_TOKENS,
                "temperature": cfg.TEMPERATURE,
                "gpu_layers": 50,
                "stream": True,
            },
        )
    else:
        llm = CTransformers(
            model=cfg.MODEL_BIN_PATH,
            model_type=cfg.MODEL_TYPE,
            batch_size=cfg.MODEL_BATCH_SIZE,
            config={
                "max_new_tokens": cfg.MAX_NEW_TOKENS,
                "temperature": cfg.TEMPERATURE,
                "stream": True,
            },
        )

    return llm

"""
===========================================
        Module: Util functions
===========================================
"""
import box
import yaml

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from src.prompts import qa_template
from pathlib import Path
from src.llm import build_llm


def get_config(config_file="config/config.yml"):
    with open("config/config.yml", "r", encoding="utf8") as ymlfile:
        return box.Box(yaml.safe_load(ymlfile))


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


def set_qa_prompt(template):
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt


def build_retrieval_qa(llm, prompt, vectordb, cfg):
    dbqa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": cfg.VECTOR_COUNT}),
        return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
        chain_type_kwargs={"prompt": prompt},
    )
    return dbqa


def vectordb(cfg):
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.EMBEDDINGS_MODEL_NAME,
        model_kwargs={"device": "cuda" if cfg.USE_GPU else "cpu"},
    )
    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)
    return vectordb


def setup_dbqa(cfg):
    index = vectordb(cfg)

    llm = build_llm(cfg)
    qa_prompt = set_qa_prompt(qa_template)
    dbqa = build_retrieval_qa(llm, qa_prompt, index, cfg)

    return dbqa

class Config:
    llm_model_name = 'ClueAI/ChatYuan-large-v2'  # 本地模型文件 or huggingface远程仓库
    embedding_model_name = 'GanymedeNil/text2vec-large-chinese'  # 检索模型文件 or huggingface远程仓库
    vector_store_path = 'resource/faiss/'
    docs_path = 'resource/txt/'
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, DirectoryLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from config import Config
from utils.AliTextSplitter import AliTextSplitter


class DocumentService(object):
    def __init__(self):

        self.config = Config.vector_store_path
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.embedding_model_name)
        self.docs_path = Config.docs_path
        self.vector_store_path = Config.vector_store_path
        self.vector_store = None

    def init_source_vector(self):
        """
        初始化本地知识库向量
        :return:
        """
        loader = DirectoryLoader(self.docs_path, glob="**/*.txt", loader_cls=TextLoader)
        # 读取文本文件
        documents = loader.load()
        text_splitter = AliTextSplitter()
        # 使用阿里的分段模型对文本进行分段
        split_text = text_splitter.split_documents(documents)
        # 采用embeding模型对文本进行向量化
        self.vector_store = FAISS.from_documents(split_text, self.embeddings)
        # 把结果存到faiss索引里面
        self.vector_store.save_local(self.vector_store_path)

    def load_vector_store(self):
        self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)



if __name__ == '__main__':
    s = DocumentService()
    ###将文本分块向量化存储起来
    s.init_source_vector()

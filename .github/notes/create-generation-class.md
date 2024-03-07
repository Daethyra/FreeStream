# Create a Generation Class
The following is a starting point for transforming the `configure_retriever` function into a class. I intend for the class to have at least a method for returning the retriever, and a method that returns an instantiated LLM(`ChatOpenAI`).

```python
class RetrieverConfigurator:
    def __init__(self, uploaded_files):
        self.uploaded_files = uploaded_files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.docs = []
        self.chunks = []
        self.vectordb = None
        self.retriever = None

    def load_documents(self):
        for file in self.uploaded_files:
            temp_filepath = os.path.join(self.temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            loader = UnstructuredFileLoader(temp_filepath)
            self.docs.extend(loader.load())
            logger.info("Loaded document: %s", file.name)

    def split_documents(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=75)
        self.chunks = text_splitter.split_documents(self.docs)

    def create_embeddings_and_store_in_vectordb(self):
        model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", model_kwargs=model_kwargs
        )
        self.vectordb = FAISS.from_documents(self.chunks, embeddings)

    def define_retriever(self):
        self.retriever = self.vectordb.as_retriever(
            search_type="mmr", search_kwargs={"k": 3, "fetch_k": 7}
        )

    def get_retriever(self):
        if not self.retriever:
            self.load_documents()
            self.split_documents()
            self.create_embeddings_and_store_in_vectordb()
            self.define_retriever()
        return self.retriever

    def instantiate_llm(self):
        # Assuming ChatOpenAI requires some configuration or initialization
        llm = ChatOpenAI()
        return llm
```
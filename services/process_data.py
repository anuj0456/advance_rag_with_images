import uuid
import base64
from unstructured.partition.pdf import partition_pdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document

class DataProcessor:
    def __init__(self):
        self.loader = PyPDFLoader("db/cpi.pdf")
        self.pdf_pages = self.loader.load()

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self.all_splits_pypdf = self.text_splitter.split_documents(self.pdf_pages)
        self.all_splits_pypdf_texts = [d.page_content for d in self.all_splits_pypdf]

    def load_images(self):
        raw_pdf_elements = partition_pdf(
            filename="db/cpi.pdf",
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            image_output_dir_path="db/",
        )

        # Categorize by type
        tables = []
        texts = []
        for element in raw_pdf_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                tables.append(str(element))
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                texts.append(str(element))

    def store_data(self):
        baseline = Chroma.from_texts(
            texts=self.all_splits_pypdf_texts,
            collection_name="baseline",
            embedding=OpenAIEmbeddings(),
        )
        retriever_baseline = baseline.as_retriever()

        def encode_image(image_path):
            """Getting the base64 string"""
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        def image_summarize(img_base64, prompt):
            """Image summary"""
            chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)

            msg = chat.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                            },
                        ]
                    )
                ]
            )
            return msg.content

    def create_multi_vector_retriever(
            self, vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
    ):
        # Initialize the storage layer
        store = InMemoryStore()
        id_key = "doc_id"

        # Create the multi-vector retriever
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )

        # Helper function to add documents to the vectorstore and docstore
        def add_documents(retriever, doc_summaries, doc_contents):
            doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
            summary_docs = [
                Document(page_content=s, metadata={id_key: doc_ids[i]})
                for i, s in enumerate(doc_summaries)
            ]
            retriever.vectorstore.add_documents(summary_docs)
            retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

        # Add texts, tables, and images
        # Check that text_summaries is not empty before adding
        if text_summaries:
            add_documents(retriever, text_summaries, texts)
        # Check that table_summaries is not empty before adding
        if table_summaries:
            add_documents(retriever, table_summaries, tables)
        # Check that image_summaries is not empty before adding
        if image_summaries:
            add_documents(retriever, image_summaries, images)

        return retriever

    def get_multivector_store(self, retriever_multi_vector_img, query, suffix_for_images):
        multi_vector_img = Chroma(
            collection_name="multi_vector_img", embedding_function=OpenAIEmbeddings()
        )
        docs = retriever_multi_vector_img.invoke(query + suffix_for_images)
        return docs


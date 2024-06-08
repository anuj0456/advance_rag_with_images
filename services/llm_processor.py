from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


class LLMProcessor:

    def __init__(self, texts=None, tables=None):
        self.texts = texts
        self.tables = tables

    def get_qtext_response(self):
        # Prompt
        prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
        These summaries will be embedded and used to retrieve the raw text or table elements. \
        Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
        prompt = ChatPromptTemplate.from_template(prompt_text)

        # Text summary chain
        model = ChatOpenAI(temperature=0, model="gpt-4")
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

        # Apply to text
        text_summaries = summarize_chain.batch(self.texts, {"max_concurrency": 5})

        # Apply to tables
        table_summaries = summarize_chain.batch(self.tables, {"max_concurrency": 5})

    def get_qimage_response(self, base64_image, image_summary):
        img_base64_list = []

        # Store image summaries
        image_summaries = []

        # Prompt
        prompt = """You are an assistant tasked with summarizing images for retrieval. \
        These summaries will be embedded and used to retrieve the raw image. \
        Give a concise summary of the image that is well optimized for retrieval."""

        # Apply to images
        for img_file in sorted("db"):
            if img_file.endswith(".jpg"):
                img_base64_list.append(base64_image)
                image_summaries.append(image_summary)

import json
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PrepareData:
    def __init__(self, json_path: str = "data/processed/parsed_data_with_descriptions.json"):
        self.json_path = json_path

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 250,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_and_chunk(self) -> List[Document]:
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = []

        for page in data:
            page_number = page["page_number"]
            source = page.get("source", "chaos_book.pdf")

            if page.get("text"):
                text_doc = Document(page_content=page["text"])
                chunks = self.text_splitter.split_documents([text_doc])

                for i, chunk in enumerate(chunks):
                    chunk.metadata = {
                        "page": page_number,
                        "source": source,
                        "type": "text",
                        "chunk_index": i
                    }
                    docs.append(chunk)

            if page.get("image_descriptions"):
                for image in page["image_descriptions"]:
                    description = image["description"]
                    path = image["path"]

                    content = f"Image description from page {page_number}: {description}"

                    image_doc = Document(
                        page_content=content,
                        metadata = {
                            "page": page_number,
                            "source": source,
                            "type": "image",
                            "image_path": path
                        }
                    )
                    docs.append(image_doc)

        return docs




            




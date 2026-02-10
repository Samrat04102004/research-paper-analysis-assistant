import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SectionAwareParser:
    def __init__(self):
        # Common academic paper section headers
        self.sections = [
            "ABSTRACT",
            "INTRODUCTION",
            "METHODOLOGY",
            "METHODS",
            "RESULTS",
            "DISCUSSION",
            "CONCLUSION",
            "REFERENCES",
            "BACKGROUND"
        ]

        # Smaller chunks prevent the LLM from getting "lost" in too much text
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

    def parse(self, file_path: str):
        doc = fitz.open(file_path)
        all_pages_docs = []
        current_section = "Introduction"

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            lines = text.split("\n")
            page_content = ""

            for line in lines:
                clean_line = line.strip().upper()

                # Detect section headers
                if any(sec in clean_line for sec in self.sections) and len(clean_line) < 20:
                    current_section = line.strip()

                page_content += line + " "

            # Create a document object for the full page
            page_doc = Document(
                page_content=page_content,
                metadata={
                    "section": current_section,
                    "page": page_num + 1,
                    "source": file_path.split("/")[-1],
                },
            )

            all_pages_docs.append(page_doc)

        # Split into smaller semantic chunks while preserving metadata
        final_chunks = self.text_splitter.split_documents(all_pages_docs)

        return final_chunks

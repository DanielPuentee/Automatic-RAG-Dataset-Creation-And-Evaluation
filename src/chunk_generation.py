import os
import pandas as pd
import PyPDF2
from langchain_text_splitters import TokenTextSplitter
import uuid

class Chunking:
    
    def __init__(self):
        pass

    def preprocess_one_document(self, path: str) -> list:
        """
        Preprocess a single document

        Args:
            path (str): path to the document

        Returns:
            list: list of paragraphs
        """
        pdf = PyPDF2.PdfReader(path)
        filename_paragraphs = []

        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            content = {
                'page': page_num + 1,
                'text': page.extract_text(),
                'filename': os.path.basename(path)
            }
            filename_paragraphs.append(content)

        return filename_paragraphs

    def preprocessing(self, paths: list) -> list:
        """
        Preprocess a list of documents

        Args:
            paths (list): list of paths to the documents

        Returns:
            list: list of paragraphs
        """
        paragraphs = []
        for path in paths:
            filename_paragraphs = self.preprocess_one_document(path)
            paragraphs += [filename_paragraphs]
        return paragraphs
    
    def chunking(self, paragraphs: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
        """
        Split the paragraphs into chunks

        Args:
            paragraphs (list): list of paragraphs
            chunk_size (int): size of the chunk
            chunk_overlap (int): overlap between chunks

        Returns:
            list: list of chunks
        """
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks_return = []

        # Itarate over all the documents
        for file_paragraphs in paragraphs:

            # Itarete over all the paragraphs in the document
            for file_paragraph in file_paragraphs:

                content = file_paragraph['text']
                list_chunks = text_splitter.split_text(content)

                filename_chunks = [{
                    'chunk_id': str(uuid.uuid4()),
                    'content': content,
                    'filename': file_paragraph['filename'],
                    'page': file_paragraph['page']
                } for content in list_chunks]

                chunks_return += filename_chunks

        return chunks_return
    
    def preprocess_chunking(self, paths: list, chunk_size: int = 500, chunk_overlap: int = 50) -> pd.DataFrame:
        """
        Preprocess and chunk the documents

        Args:
            paths (list): list of paths to the documents
            chunk_size (int): size of the chunk
            chunk_overlap (int): overlap between chunks

        Returns:
            pd.DataFrame: dataframe of chunks
        """
        paragraphs = self.preprocessing(paths)
        chunks = self.chunking(paragraphs, chunk_size, chunk_overlap)
        df_chunks = pd.DataFrame(chunks)
        return df_chunks
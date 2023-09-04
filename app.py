import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS



def get_pdf_text(pdf_files):
    text = ""
    for pdf in  pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(chunks_text):
    embdeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=chunks_text, embedding=embdeddings)
    return vector_store

def main():
    load_dotenv()
    st.set_page_config(page_title="PDFConvo", page_icon=":books:")

    st.header("PDFConvo :books:")
    st.text_input("Ask a question about yout document:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_files = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_files)
                # st.write(raw_text)

                chunks_text = get_chunks(raw_text)
                # st.write(chunks_text)

                vector_store = get_vector_store(chunks_text)




if __name__ == '__main__':
    main()
import os
import argparse
from langchain_community.vectorstores import DeepLake
from utils import convert_files_to_txt, convert_website_to_text
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()
# os.environ['ACTIVELOOP_TOKEN'] = 'eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpZCI6ImFhamFpcyIsImFwaV9rZXkiOiJoVjBoU0JIVk1OM25kV05ocEdHV2NFRVZiOThNbjBIUk5SS1dfNEF3WTNrLU0ifQ.'

def run(args):
    if args.flag == 'website':
        urls = ["https://docs.dipy.org/stable/examples_built/", "https://docs.dipy.org/stable/reference/index.html", "https://docs.dipy.org/stable/interfaces/", "https://docs.dipy.org/stable/reference_cmd/", "https://github.com/dipy/dipy/discussions"]
        texts_split = convert_website_to_text(urls)
    elif args.flag == 'source':
        path = str(input("Enter codebase filepath: "))
        texts_split = convert_files_to_txt(src_dir=path, dst_dir='converted_codebase')

    # Nomic v1 or v1.5
    embd_model_path = r"model\nomic-embed-text-v1.5.Q5_K_S.gguf"
    embedding = LlamaCppEmbeddings(model_path=embd_model_path, n_batch=512)

    if args.upload:
        username='aajais'
        db = DeepLake.from_documents(texts_split, dataset_path=f"hub://{username}/dipy-v2", embedding=embedding)
        retriever = db.as_retriever()
        return retriever
    else:
        # Index
        vectorstore = Chroma.from_documents(
            documents=texts_split,
            collection_name="rag-chroma",
            embedding=embedding,
        )
        retriever = vectorstore.as_retriever()
        return retriever

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--flag', type=str, help='flag for either website or source')
    parser.add_argument('--upload', default=False, action="store_true", help='upload data to cloud')
    args = parser.parse_args()
    retriever = run(args)
    print(retriever)
import os
import tiktoken
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as Soup


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def convert_files_to_txt(src_dir, dst_dir):
    # If the destination directory does not exist, create it.
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py') or file.endswith('.pyx') or file.endswith('.build') or file.endswith('.ipynb') or file.endswith('.rst') or file.endswith('.sh') or file.endswith('.txt'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, src_dir)
                # get the relative path to preserve directory structure
                # Create the same directory structure in the new directory
                new_root = os.path.join(dst_dir, os.path.dirname(rel_path))
                os.makedirs(new_root, exist_ok=True)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            data = f.read()
                    except UnicodeDecodeError:
                        print(f"Failed to decode the file: {file_path}")
                        continue
                # Create a new file path with .txt extension
                new_file_path = os.path.join(new_root, file + '.txt')
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    f.write(data)

    new_src_dir = os.path.join(os.getcwd(), dst_dir)
    loader = DirectoryLoader(new_src_dir, show_progress=True, loader_cls=lambda path: TextLoader(path, encoding='utf-8'))
    repo_files = loader.load()
    print(f"Number of files loaded: {len(repo_files)}")
    #
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1712, chunk_overlap=500)
    documents = text_splitter.split_documents(documents=repo_files)
    print(f"Number of documents : {len(documents)}")

    return documents


def convert_website_to_text(urls):
    def load_docs(url, max_depth):
        loader = RecursiveUrlLoader(
            url=url, max_depth=max_depth, extractor=lambda x: Soup(x, "html.parser").text
        )
        return loader.load()

    docs = []
    for url in urls:
        docs_url = load_docs(url, 2)
        docs.extend(*docs_url)
    docs_texts = [d.page_content for d in docs]

    # Calculate the number of tokens for each document
    counts = [num_tokens_from_string(d, "cl100k_base") for d in docs_texts]

    # Plotting the histogram of token counts
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=30, color="blue", edgecolor="black", alpha=0.7)
    plt.title("Histogram of Token Counts")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.show()

    # Concatenate documents
    d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
    d_reversed = list(reversed(d_sorted))
    concatenated_content = "\n\n\n --- \n\n\n".join(
        [doc.page_content for doc in d_reversed]
    )
    num_tokens = num_tokens_from_string(concatenated_content, "cl100k_base")
    print(f"Num tokens in all context: {num_tokens}")

    # Split concatenated content into chunks
    chunk_size_tok = 2000
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tok, chunk_overlap=0
    )
    texts_split = text_splitter.split_documents(concatenated_content)

    return texts_split


if __name__=="__main__":
    path = str(input("Enter codebase filepath: "))
    # Call the function with the source and destination directory paths
    convert_files_to_txt(path, 'converted_codebase')

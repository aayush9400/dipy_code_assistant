import matplotlib.pyplot as plt
import tiktoken
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores import Chroma


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# DIPY API
url = "https://docs.dipy.org/stable/reference/index.html"
loader = RecursiveUrlLoader(
    url=url, max_depth=10000, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

# DIPY Tutorials
url = "https://docs.dipy.org/stable/examples_built/"
loader = RecursiveUrlLoader(
    url=url, max_depth=10000, extractor=lambda x: Soup(x, "html.parser").text
)
docs_tutorials = loader.load()

# DIPY CLI Workflows
url = "https://docs.dipy.org/stable/interfaces/"
loader = RecursiveUrlLoader(
    url=url, max_depth=10000, extractor=lambda x: Soup(x, "html.parser").text
)
docs_workflows = loader.load()

# DIPY CLI API
url = "https://docs.dipy.org/stable/reference_cmd/"
loader = RecursiveUrlLoader(
    url=url, max_depth=10000, extractor=lambda x: Soup(x, "html.parser").text
)
docs_cli_api = loader.load()

# DIPY Discussions
url = "https://github.com/dipy/dipy/discussions"
loader = RecursiveUrlLoader(
    url=url, max_depth=10000, extractor=lambda x: Soup(x, "html.parser").text
)
docs_discuss = loader.load()

# Doc texts
docs.extend([*docs_tutorials, *docs_workflows, *docs_cli_api, *docs_discuss])
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

# Display the histogram
plt.show()

# Doc texts concat
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)
print(
    "Num tokens in all context: %s"
    % num_tokens_from_string(concatenated_content, "cl100k_base")
)

# Doc texts split
chunk_size_tok = 2000
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=chunk_size_tok, chunk_overlap=0
)
texts_split = text_splitter.split_documents(concatenated_content)


# Nomic v1 or v1.5
embd_model_path = "/home/aajais/Desktop/DiPyCodeAssistant/llama.cpp/models/nomic-embed-text-v1.5.Q5_K_S.gguf"
embedding = LlamaCppEmbeddings(model_path=embd_model_path, n_batch=512)

# Index
vectorstore = Chroma.from_documents(
    documents=texts_split,
    collection_name="rag-chroma",
    embedding=embedding,
)
retriever = vectorstore.as_retriever()
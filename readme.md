# Project: DIPY Code Assistant

## Overview

The DIPY Code Assistant is a comprehensive tool designed to facilitate the extraction, processing, and retrieval of information from the DIPY documentation and codebase. Utilizing advanced language models and embedding techniques, this tool aims to provide an efficient and user-friendly interface for interacting with the extensive documentation of DIPY, a leading library for diffusion MRI and related processing.

## Features

- **Document Conversion**: Convert various source files and websites into text format, ensuring compatibility and ease of processing.
- **Embeddings and Vector Stores**: Employ LlamaCppEmbeddings and DeepLake vector stores for efficient information retrieval.
- **Question Answering**: Implement a QA system to answer user queries based on the extracted documents.
- **Streamlit Interface**: Provide an interactive web application for user interaction.

## Prerequisites

- Python 3.8+
- Streamlit
- Langchain Community
- Matplotlib
- BeautifulSoup
- Tiktoken
- dotenv
- DeepLake

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/dipy-code-assistant.git
   cd dipy-code-assistant
   ```

2. **Create a Virtual Environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```
3. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    Set Up Environment Variables:
    ```

4. **Set Up Environment Variables**: \
*Create a .env file in the project root directory and add your configuration details*.

    ```ACTIVELOOP_TOKEN=your_deeplake_token```
    
## Usage

### Streamlit Application

1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
    ```
2. **Interact with the Application**:
Open your browser and navigate to the displayed URL (default: http://localhost:8501).
Enter your queries in the text input box and receive responses from the bot.

### Command Line Interface

1. **Convert Codebase Files to Text**:
   ```bash
   python preprocess.py --flag source --upload
   ```

2. **Convert Website Content to Text**:

    ```bash
    python preprocess.py --flag website --upload
    ```

## Project Structure
dipy-code-assistant \
├── app.py # Streamlit application\
├── create_db.py # Script to create the database and retriever \
├── preprocess.py # Script to preprocess files and websites \
├── utils.py # Utility functions for processing\
├── requirements.txt # List of dependencies\
├── .env # Environment variables\
├── README.md # Project README file\
└── model/ # Directory to store model files
## Functions and Modules

### app.py
- `load_model`: Caches and loads the language model.
- `load_retriever`: Caches and loads the document retriever.
- `handle_qa`: Handles user queries and provides answers using the loaded model and retriever.
- **Streamlit UI Setup**: Sets up the user interface for interaction.

### create_db.py
- `run`: Main function to process input and create a retriever based on the provided flag (source or website).

### preprocess.py
- `convert_files_to_txt`: Converts source code files to text format and splits them into chunks.
- `convert_website_to_text`: Converts website content to text format and splits it into chunks.

### utils.py
- `num_tokens_from_string`: Calculates the number of tokens in a text string.
- `RecursiveCharacterTextSplitter`: Splits documents into chunks for efficient processing.

## Future Enhancements

- **Enhanced Error Handling**: Improve error handling mechanisms for robust processing.
- **Additional Embedding Models**: Integrate more embedding models to support a wider range of document types.
- **User Authentication**: Implement user authentication for secure access to the tool.
- **Advanced Search Capabilities**: Develop more sophisticated search algorithms to provide better and faster results.
- **Integration with Other Data Sources**: Enable the tool to retrieve and process information from additional data sources and APIs.
- **User Interface Improvements**: Enhance the Streamlit interface for better user experience and accessibility.
- **Performance Optimization**: Optimize the tool's performance for faster processing and response times.
- **Detailed Logging and Monitoring**: Implement logging and monitoring features to track the tool's usage and performance metrics.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue on the GitHub repository.

## License

This project is licensed under the MIT License. See the [MIT License](LICENSE) file for more details.



## Acknowledgments

- The DIPY development team for their extensive documentation and support.
- The Langchain Community for their powerful tools and libraries.
- The OpenAI community for continuous support and development of state-of-the-art models.
- The contributors and maintainers of the open-source projects used in this tool.
- The users and testers who provided valuable feedback and suggestions.

## Contact

For any queries or support, please reach out to [Aayush Jaiswal](https://linkedin.com/in/jaiswal-aayush).
import os
from setuptools import setup

# Set environment variables
os.environ["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=on"
os.environ["FORCE_CMAKE"] = "1"

# Run the pip install command
os.system("pip install llama-cpp-python")

setup(
    name="your_project_name",
    version="0.1",
    install_requires=[
        'llama-cpp-python'
    ],
)

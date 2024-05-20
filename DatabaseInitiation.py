# This script is used to load the csv file into a Chroma database.

from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

REVIEWS_CSV_PATH = "detailed_fictional_impressive_people.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

key = ''
loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="Description")
reviews = loader.load()

reviews_vector_db = Chroma.from_documents(
    reviews, OpenAIEmbeddings(openai_api_key = key), persist_directory=REVIEWS_CHROMA_PATH
)
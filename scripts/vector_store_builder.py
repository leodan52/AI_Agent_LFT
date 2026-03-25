from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from models.vector_storage import VectorStorage


def main():

	model = "models/gemini-embedding-001"
	data_path = "data/"

	# Load API KEY

	load_dotenv()

	# Gemini embeddings model

	embeddings = GoogleGenerativeAIEmbeddings(
		model=model
	)

	# VectorStorage instance

	storage = VectorStorage(embeddings)
	storage.generate_embeddings(data_path, chunk_size=500, chunk_overlap=50, rpm=100)

	# Vector store persistence

	storage.store_vectors("db/")


if __name__ == "__main__":
	main()
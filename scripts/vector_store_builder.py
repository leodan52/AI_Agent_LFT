import json
import os
from os.path import isfile

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from models import vector_storage
from models.vector_storage import VectorStorage


def main():
	settings_path = "settings/agent.json"

	if os.path.isfile(settings_path):
		with open(settings_path, "r") as f:
			settings = json.load(f)
	else:
		settings = {
			"model" : "models/gemini-embedding-001",
			"data_path" : "data/",
			"vector_store_path" : "vector_store",
		}


	# Load API KEY

	load_dotenv()

	# Gemini embeddings model

	embeddings = GoogleGenerativeAIEmbeddings(
		model=settings["model"]
	)

	# VectorStorage instance

	storage = VectorStorage(embeddings)
	storage.generate_embeddings(
		settings["data_path"], chunk_size=1000, chunk_overlap=100
	)

	# Vector store persistence

	storage.store_vectors(settings["vector_store_path"])


if __name__ == "__main__":
	main()
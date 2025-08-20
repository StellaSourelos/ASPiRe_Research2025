# GAN (Stella project idea)
# How to run:

-Create a virtual environment:
	python3.10 -m venv rag-env

-Activate/Reactivate the environment:
	source rag-env/bin/activate

-Install required Python files:
	pip install flask transformers pandas scikit-learn torch nltk
	pip install -U sentence-transformers

-Build DB:
	sqlite3 knowledge.db < schema.sql
	sqlite3 knowledge.db < sample_data.sql

-Run the RAG server:
	python proto3.py

-Open your browser:
	http://127.0.0.1:5000/

	

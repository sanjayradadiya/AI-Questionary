import os

from flask import Flask, request
from flask_cors import CORS 
from werkzeug.utils import secure_filename
from langchain import OpenAI, VectorDBQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

# Configuration for file upload folder
app.config['FILES'] = "files"

ALLOWED_EXTENSIONS = {'pdf'}

def validate_file_extensions(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/intelligent_automation", methods=["POST"])
def file_upload():
    file_paths = []
    params = request.form
    if params is None:
        params = request.json
        if params is None:
            return "Parameters missing", 400
        
    if request.files:
        for file_path in request.files.keys():
            path = request.files[file_path]
            if path and validate_file_extensions(path.filename):
                file_path = os.path.join( app.config['FILES'], secure_filename(path.filename))
                path.save(file_path)
                file_paths.append(file_path)
    
    if len(file_paths) == 0:
        return "Missing file paths", 400

    questions = []
    if "questions" in params:
        questions = params["questions"]
    else:
        return "Questions param is missing", 400

    results = []
    for file_path in file_paths:
        file_response = file_parse(file_path, questions)
        results.append(file_response['file_key'])

    return results

def file_parse(file_path, questions):
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size = 1500, chunk_overlap = 200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    vector_store = Chroma.from_documents(texts, embeddings)
    
    vector_chain = VectorDBQA.from_chain_type(llm = OpenAI(), chain_type = "stuff", vectorstore = vector_store)
    
    answer_result = []
    for question in questions:
        answer = vector_chain.run(question)
        answer_result.append({
            "question": question,
            "answer": answer
            })
    
    return {
        "file": file_path,
        "answers": answer_result,
        }

if __name__ == '__main__': 
   app.run()


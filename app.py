import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

app = Flask(__name__)
CORS(app)

# --- Load your resume text (from ./data/*.txt) ---
#DOC_FILES = ["data/resume.txt", "data/projects.txt", "data/skills.txt"]
DOC_FILES = ["data/contact.txt", "data/resume.txt", "data/projects.txt", "data/skills.txt"]

docs_text = ""
for p in DOC_FILES:
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            docs_text += f.read().strip() + "\n"

# Fallback if you haven't created the files yet:
if not docs_text.strip():
    docs_text = """
    15+ years in Oracle EPM (PBCS, FCCS, Essbase). Leads Humana architecture and production support.
    Python, Flask, REST APIs, Flutter. Agentic AI with LangChain/LangGraph. Clients: CAP, Citizens, RBS, ICA, Baker Hughes.
    Upgrades: 11.1.2.2 -> 11.1.2.4; FDMEE/CDM/OIC integrations; CAB/UAT/change mgmt.
    """

# --- Build retriever ---
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
chunks = splitter.split_text(docs_text)

# Requires: export OPENAI_API_KEY=your_key
emb = OpenAIEmbeddings()
vs = FAISS.from_texts(chunks, emb)
retriever = vs.as_retriever(search_kwargs={"k": 4})
llm = ChatOpenAI(model="gpt-4o-mini")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True)

@app.post("/ask")
def ask():
    try:
        q = (request.get_json(force=True).get("question") or "").strip()
    except Exception:
        q = ""
    if not q:
        return jsonify({"answer": "Please provide a question."}), 400
    try:
        result = qa.invoke({"query": q})
        answer = result.get("result", "No answer.")
        refs = [d.page_content[:150] for d in result.get("source_documents", [])]
        return jsonify({"answer": answer, "refs": refs})
    except Exception as e:
        return jsonify({"answer": f"Agent unavailable. ({e})"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))

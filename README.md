# 🔍 Retrieval-Augmented Generation (RAG) Pipeline  
A complete end-to-end RAG system built using **LangChain**, **OpenAI models**, **Whisper**, and **vector databases**. This project demonstrates how to convert raw audio/text data into structured embeddings, store them in a vector store, retrieve relevant chunks using semantic similarity, and generate context-aware answers using an LLM.

---

## 🚀 Features

### **1. Audio-to-Text Pipeline (Whisper)**
- Downloads YouTube videos using `pytube`.
- Converts audio to text using **Whisper**, producing accurate transcripts.
- Supports large audio files, writing full transcription to disk.

### **2. Document Chunking & Preprocessing**
- Uses **RecursiveCharacterTextSplitter** to break transcripts into overlapping chunks.
- Optimized for RAG workflows (chunk_size = 1000, overlap = 20).
- Maintains semantic continuity across chunks.

### **3. Embedding Generation**
- Generates high-quality embeddings using **OpenAIEmbeddings**.
- Embeds:
  - Raw text
  - Queries
  - YouTube-transcribed documents
- Also includes cosine similarity comparisons using **scikit-learn**.

### **4. Vector Stores**
Supports multiple vector database backends:
- **DocArrayInMemorySearch** (in-memory indexing)
- **PineconeVectorStore** (production scalable index)

Features:
- Similarity search  
- Similarity scores  
- Retriever interface  

### **5. Full RAG Chain (Context + LLM)**
Built using:
- `ChatPromptTemplate`
- `RunnableParallel`
- `RunnablePassthrough`
- `StrOutputParser`
- OpenAI Chat Models (GPT-3.5/GPT-4 versions)

Pipeline:
1. User question →  
2. Query embedding →  
3. Vector store retrieval →  
4. Context assembly →  
5. LLM generates factual, grounded response  

---

## 🛠️ Technologies Used

| Component | Technology |
|----------|------------|
| LLM | OpenAI GPT-3.5 / GPT-4 |
| Framework | LangChain |
| Embeddings | OpenAIEmbeddings |
| Vector DB | DocArray, Pinecone |
| STT | OpenAI Whisper |
| Retrieval | Semantic Search, cosine similarity |
| Languages | Python |
| Tools | pytube, sklearn, json, dotenv |

---

## 🧠 How It Works (High-Level Flow)

1. **Audio → Text**  
   Convert YouTube videos to text via Whisper.

2. **Text Processing**  
   Clean, split, and prepare documents for indexing.

3. **Embedding + Indexing**  
   Store text embeddings in DocArray or Pinecone.

4. **Query Handling**  
   Convert user input to embeddings.

5. **Semantic Retrieval**  
   Fetch top-k relevant chunks using vector search.

6. **LLM Response Generation**  
   Pass context + question through an LLM chain → accurate answer.

---

## 📌 Use Cases

✔ Creating personal AI assistants  
✔ RAG systems for long audio/video content  
✔ YouTube lecture summarization  
✔ Custom knowledge base question-answering  
✔ Semantic search engine prototypes  

---


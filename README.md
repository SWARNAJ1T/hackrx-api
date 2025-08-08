# 🛡️ HackRx Insurance Policy QA API (RAG + LLM)

This project is a **Question Answering API** for **insurance policies**, built using a **Retrieval-Augmented Generation (RAG)** pipeline. It reads policy PDFs, embeds the content, and uses a local LLM to generate **step-by-step, clause-grounded responses**.

## 🔧 Features

- 📄 **Ingests PDFs from URL**
- 🧠 **Embeds text using `sentence-transformers`**
- 📦 **Stores vectors in Pinecone for retrieval**
- 🤖 **Uses a local LLM (`DeepSeek-R1-Distill-Qwen-1.5B`) for generation**
- 🪜 **Outputs structured responses with step-by-step reasoning**
- 🚀 **FastAPI-powered API with a `/run` endpoint**

---

## 📌 Example Input (POST `/api/v1/hackrx/run`)

```json
{
  "documents": [
    "https://example.com/sample-policy.pdf"
  ],
  "questions": [
    "What is the grace period for premium payment?",
    "Is OPD dental treatment covered?"
  ]
}

# personalisation_proj# ShopSmart: Hybrid Two-Tower Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.9-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-Serving-green)

**ShopSmart** is a production-grade recommendation service capable of serving personalized product rankings in real-time. It utilizes a **Two-Tower Neural Network architecture** to generate embeddings for users and items, allowing for efficient semantic retrieval and ranking.

## 🏗 Architecture
The system follows a standard retrieval-ranking pipeline:
1.  **Offline Training:** A PyTorch Two-Tower model is trained on interaction data (clicks/purchases).
2.  **Indexing:** Item embeddings are indexed using **FAISS** for millisecond-latency approximate nearest neighbor search.
3.  **Serving:** A FastAPI endpoint accepts a `user_id`, generates the user embedding on the fly, and queries the index.

## 🚀 Key Features
* **Two-Tower Architecture:** Separate deep neural networks for User and Item features.
* **Hybrid Filtering:** Combines interaction history (Collaborative) with metadata (Content-based).
* **Low-Latency API:** Built with FastAPI and optimized for sub-50ms response times.
* **Dockerized:** Fully containerized for easy deployment.

## 🛠 Installation
```bash
git clone [https://github.com/yourusername/shopsmart-recsys.git](https://github.com/yourusername/shopsmart-recsys.git)
cd shopsmart-recsys
pip install -r requirements.txt
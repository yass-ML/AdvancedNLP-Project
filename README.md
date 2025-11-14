# Few-Shot Learning Optimization: Finding the Optimal Prompt Strategy

## 📌 Overview

This project investigates how the number and type of few-shot examples influence the performance of small language models on classification tasks. We systematically compare different example selection strategies across multiple shot counts to identify the optimal balance between performance, annotation cost, and inference efficiency.

Our work is inspired by the paper **"The Few-shot Dilemma: Over-prompting Large Language Models"** ([arXiv:2509.13196](https://arxiv.org/pdf/2509.13196)), which highlights the trade-offs in few-shot prompting for large language models.

---

## 🎯 Motivation

Few-shot prompting enables language models to perform tasks with minimal training examples. However, the relationship between the number of examples and model performance is not straightforward. Adding more examples can improve accuracy, but it also increases computational costs and may lead to diminishing returns or even performance degradation.

This project explores the **few-shot dilemma** for smaller language models (1B–3B parameters) and seeks to answer:

- How does the number of examples (0, 1, 3, 5, 10, 20) affect classification performance?
- Which example selection strategy (random, semantic similarity, TF-IDF) works best?
- How do these effects vary across different model sizes and tasks?

---

## 🧪 Methodology

### Models

We evaluate the following small language models:

- **TinyLlama** (1.1B parameters)
- **Llama 3.2** (1B and 3B variants)
- **Phi-3-mini** (3.8B parameters)

### Tasks

We focus on classification tasks, including:

- **Sentiment Analysis**: Classifying text as positive, negative, or neutral.
- **Named Entity Recognition (NER)**: Identifying and categorizing entities in text.
- **Intent Classification**: Determining user intent from conversational input.

### Selection Strategies

We compare three methods for selecting few-shot examples:

1. **Random Selection**: Examples chosen uniformly at random.
2. **Semantic Similarity**: Examples selected based on cosine similarity to the input query (using sentence embeddings).
3. **TF-IDF Similarity**: Examples selected based on TF-IDF vector similarity to the input.

### Baselines

To contextualize our findings, we include:

- **Zero-shot performance**: Models evaluated without any examples.
- **Supervised fine-tuned encoder**: A baseline using a fine-tuned BERT-style encoder for classification.

---

## 📊 Evaluation

We measure performance using standard classification metrics:

- **Accuracy**
- **F1-score** (macro-averaged)
- **Inference time** (to quantify computational cost)

Each configuration is evaluated across multiple runs to ensure statistical reliability. Results are aggregated to map the performance-cost trade-off and identify the optimal prompting strategy for each model and task.

---

## 🚀 Expected Contributions

This project aims to:

1. **Quantify the few-shot dilemma** for small language models across different tasks and shot counts.
2. **Compare example selection strategies** to determine which approach maximizes accuracy with minimal examples.
3. **Provide practical recommendations** for researchers and practitioners on effective few-shot prompting strategies.
4. **Establish performance baselines** using zero-shot and fine-tuned supervised models for comparison.

---

## 📚 References

- **The Few-shot Dilemma: Over-prompting Large Language Models**  
  [https://arxiv.org/pdf/2509.13196](https://arxiv.org/pdf/2509.13196)

---

## 👥 Authors

* Cédric Damais
* Léon Ayral
* Amine Mike El-Maalouf
* Yacine Benihaddadene


---

## 📄 License

This project is developed as part of the Advanced NLP course in the SCIA Major at EPITA - École pour l'informatique et les techniques avancées.
It is provided for educational and research purposes.



# 🎬 Quick RAG Evaluation with TV Series Q&A  

*A lightweight Retrieval-Augmented Generation (RAG) pipeline using Langchain, RAGAS, Giskard, Gemmini & LangSmith*

![GitHub top language](https://img.shields.io/github/languages/top/DanielPuentee/Automatic-RAG-Dataset-Creation-And-Evaluation)
[![Visual Studio Code](https://custom-icon-badges.demolab.com/badge/Visual%20Studio%20Code-0078d7.svg?logo=vsc&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![Anaconda](https://img.shields.io/badge/Anaconda-44A833?logo=anaconda&logoColor=fff)](#)
![GitHub last commit](https://img.shields.io/github/last-commit/DanielPuentee/Automatic-RAG-Dataset-Creation-And-Evaluation)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub Copilot](https://img.shields.io/badge/GitHub%20Copilot-000?logo=githubcopilot&logoColor=fff)](#)

> Author: [Daniel Puente Viejo](https://www.linkedin.com/in/danielpuenteviejo/)

## 🎯 Objective

This repository demonstrates how to **quickly evaluate RAG systems** without the need to manually create a large dataset.  
We use a sample use case: **answering questions about popular TV series** like *Breaking Bad* and *La Casa de Papel*.  
The pipeline is fully open-source and built using:

- [Langchain](https://www.langchain.com/)
- [Gemmini](https://github.com/langchain-ai/gemmini) – a completely open-source orchestration framework for LLM apps
- [RAGAS](https://github.com/explodinggradients/ragas) – for evaluating RAG responses
- [Giskard](https://giskard.ai/) – to detect hallucinations, bias, and robustness issues
- [LangSmith](https://smith.langchain.com/) – to monitor, debug, and evaluate LLM usage at scale

---

## 🧠 Use Case

We simulate a real-world scenario:
> A user asks detailed questions about a TV show, such as character arcs, plot developments, or ethical decisions.  
> The system retrieves summaries of episodes and returns a relevant, accurate response.

---

## 🛠️ Tools Used

| Tool        | Role |
|-------------|------|
| **Langchain** | Build the RAG pipeline (retriever + LLM) |
| **Gemmini** | Open-source LLM orchestration & agent management |
| **RAGAS** | Automatically evaluate generated answers |
| **Giskard** | Test model outputs for hallucinations, bias, robustness |
| **LangSmith** | Monitor and log RAG chains and metrics at runtime |

---

## 🧪 Evaluation Strategy

We eliminate the need to create a labeled dataset from scratch by:

1. Generating **realistic questions** (e.g. from critics, students, or fans)
2. Feeding them through a **Langchain RAG pipeline**
3. Using **RAGAS** to compute evaluation metrics:
   - Context Precision
   - Faithfulness
   - Answer Relevancy
4. Running **Giskard** tests to detect:
   - Hallucinations
   - Sensitivity to question changes
   - Ethical issues or biases
5. Tracking all generations and context chunks using **LangSmith**

---

## 📂 Notebook Structure

```text
1. 🔧 Setup
   - Install and import dependencies
   - Load and chunk source documents (TV series summaries)
   - Configure retriever and LLM using Gemmini

2. ❓ Generate Realistic Questions
   - Prompt templates simulate questions from fans, critics, or students

3. 🔄 Run the RAG Pipeline
   - Use Langchain to retrieve context and generate answers

4. 📊 Evaluate Responses
   - Use RAGAS to compute metrics
   - Use Giskard to run test suites

5. 📈 Monitor & Debug
   - Use LangSmith to trace and evaluate every generation

6. ✅ Summary
   - Discuss quick insights and next steps

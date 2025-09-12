# FreeStream

Streamlit Chatbots - Work in Progress

***TLDR***:
- Build on top of an already robust chatbot that reasons before responding, and has access to tools
- Customize a chatbot to respond exactly how you want
- Hostable for free through Streamlit Community Cloud
- API keys required
- Pay-per-use ChatGPT style interface

## Table of Contents

- [Quickstart](#quickstart)
  - [Installation](#installation)
- [Description](#description)
- [License](./LICENSE)
- [LLM Providers' Privacy Policies](#llm-providers-privacy-policies)

## Quickstart

This app is hosted via Streamlit Community Cloud, [here](https://freestream.streamlit.app/ "Current Version: 4.0.1")

### Installation

This project uses `poetry` for dependency management to be consistent with Streamlit Community Cloud's deployment process.

Install `poetry` with:
```bash
pip install -U pip && pip install -U poetry
```

Then, install the project's dependencies in a virtual environment using `poetry`. 

Run:

```bash
poetry install
```

You will need to set all required secrets(API keys), which require their own respective accounts.
Make a copy of "template.secrets.toml" and rename it to "secrets.toml" in the root of the project. Fill out each field in the file.


You can then start the development server with hot reloading by running:

```bash
poetry run streamlit run ./freestream/🏡_Home.py
```

---

## Description
Just a freaking robot.

*Here's some papers I found interesting when first learning about generative AI and "augmented generation."*
| **Concept** | **Definition** |
| ---- | ---------- |
| [Large Language Model](https://en.wikipedia.org/wiki/Large_language_model "Wikipedia: Large language model") | A model that can generate text. |
| [RAG](https://arxiv.org/abs/2005.11401 "Arxiv: 2005.11401") | Retrieval Augmented Generation |
| [C-RAG](https://arxiv.org/abs/2401.15884 "Arxiv: 2401.15884") | Corrective-Retrieval Augmented Generation |
| [Self-RAG](https://arxiv.org/abs/2310.11511 "Arxiv: 2310.11511") | Self-reflective Retrieval Augmented Generation |
| [ColBERT](https://arxiv.org/abs/2004.12832 "Arxiv: 2004.12832") | Efficient BERT-Based Document Search |
| [RAPTOR](https://arxiv.org/abs/2401.18059 "Arxiv: 2401.18059") | Recursive Abstractive Processing for Tree-Organized Retrieval |

---

# [License](./LICENSE)

# LLM Providers' Privacy Policies

- [DeepSeek Privacy Policy](https://cdn.deepseek.com/policies/en-US/deepseek-privacy-policy.html)
- [Streamlit](https://streamlit.io/privacy-policy/)


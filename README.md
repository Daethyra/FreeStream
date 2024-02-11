# FreeStream

Providing AI solutions for everyday people

***TLDR***:
- Unlimited file upload per user-session (deletes data on exiting the web page)
- Free access to generative AI models
- Leverage the latest techniques for accurate, helpful text generation without managing the underlying prompt-engineering

## Table of Contents

- [Quickstart](#quickstart)
  - [Installation](#installation)
- [Description](#description)
  - [Vocabulary](#critical-vocabulary)
  - [Current Functionality](#what-can-freestream-do-for-me-currently)
- [Future Functionality Plans](#future-functionality-plans)

## Quickstart

As of version 1.0.0, a test version is hosted via Streamlit Community Cloud, [here](https://freestream.streamlit.app/ "Link")

### Installation

This project uses `poetry` for dependency management because that's what Streamlit Community Cloud uses to deploy the project.

To install the project's dependencies in a virtual environment using poetry, run:

```bash
poetry install
```

You can then start the development server with hot reloading by running:

```bash
poetry run streamlit run ./freestream/main.py
```

---

## Description
The original inspiration for this project was to create a chatbot for friends in law and medicine, but I quickly realized the system should be flexible enough to serve in any domain.

#### -- **Critical Vocabulary** --

| **Vocab** | **Definition** |
| ---- | ---------- |
| RAG | Retrieval Augmented Generation |
| C-RAG | Corrective-Retrieval Augmented Generation |
| Self-RAG | Self-reflective Retrieval Augmented Generation |

### What can FreeStream do for me, currently?

Right now, FreeStream is basically a chatbot powered by GPT-3.5-Turbo that requires that you upload a file(s) before you interact with it. You're taking advantage of prompt-engineering logical flow that helps ensure the best results are retrieved from your uploaded files.

#### Things worth noting:
- Currently only supports the GPT-3.5-Turbo model
- The implemented `qa_chain` forces answers to be based on the context, and the context only. This makes it difficult to interact with the chat history in a nuanced, meaningful way.

## Future Functionality Plans

- [x] Create an RAG chatbot
- [ ] Add Gemini-Pro to the model list
- [ ] Add AI decision making
  - [ ] Implement Corrective-RAG
- [ ] Turn into a Multi-Page Application (MPA)
  - [ ] (Homepage) Add a welcome screen with a description and table of contents
  - [ ] (Page) Migrate RAG SPA code
  - [ ] (Page) Add habit tracking spreadsheet template with visualization feature
  - [ ] (Page) Add a "Task Transcriber" page that transcribes the user's speech into a task outlined with details and requirements, along with action steps to enact the plan

---

[License](./LICENSE)

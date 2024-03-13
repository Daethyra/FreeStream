# FreeStream

Try out different AI tools.

***TLDR***:
- Free
- General purpose AI chatbots with the ability to read files
  - Leverages retrieval techniques to improve response quality 
- Your data is deleted upon exit
- No sign-up necessary

## Table of Contents

- [Quickstart](#quickstart)
  - [Installation](#installation)
- [Description](#description)
  - [Vocabulary](#critical-vocabulary)
  - [Current Functionality](#what-can-freestream-do-for-me-currently)
- [Functional Requirements](#functional-requirements)
- [Non-Functional Requirements](#non-functional-requirements)
- [Roadmap](#roadmap)
  - [Thinking Out Loud](#thinking-out-loud) 
  - [Future Functionality Plans](#future-functionality-plans)
- [License](./LICENSE)
- [Privacy Policy](#privacy-policy)

## Quickstart

This app is hosted via Streamlit Community Cloud, [here](https://freestream.streamlit.app/ "Current Version: 3.0.0")

### Installation

This project uses `poetry` for dependency management because that's what Streamlit Community Cloud uses to deploy the project.

Install it with:
```bash
pip install -U pip && pip install -U poetry
```

Then, install the project's dependencies in a virtual environment using poetry. 

Run:

```bash
poetry install
```

You will need to set all required secrets, which require their own respective accounts.
Make a copy of "template.secrets.toml" and rename it to "secrets.toml" in the root of the project. Fill out each field in the file.

**Need API Keys?**
| **API Platform** | **Link** |
| ---- | ---------- |
| OpenAI | https://platform.openai.com/api-keys |
| Langchain | https://smith.langchain.com/ |
| Google | https://aistudio.google.com/app/apikey |
| Claude | https://console.anthropic.com/ |

You can then start the development server with hot reloading by running:

```bash
poetry run streamlit run ./freestream/🏡_Home.py
```

---

## Description
I originally created this project as a chatbot for law and medical professionals, but I quickly realized a more flexible system would benefit everyone.

#### -- **Important Vocabulary** --

| **Vocab** | **Definition** |
| ---- | ---------- |
| [RAG](https://arxiv.org/abs/2005.11401 "Arxiv: 2005.11401") | Retrieval Augmented Generation |
| [C-RAG](https://arxiv.org/abs/2401.15884 "Arxiv: 2401.15884") | Corrective-Retrieval Augmented Generation |
| [Self-RAG](https://arxiv.org/abs/2310.11511 "Arxiv: 2310.11511") | Self-reflective Retrieval Augmented Generation |

### What can FreeStream do for me, currently?

FreeStream functions as a chatbot powered by GPT-3.5-Turbo or Gemini-Pro. Upload a file (or files) and take advantage of advanced prompt engineering for the most helpful results based on your uploaded content. It additionally has an image upscaler that can upscale to arbitrary ratios.

**Note**:
* The implemented RAG logic strictly adheres to the uploaded context. This limits meaningful interaction with the chat history.

## Functional Requirements

The application **MUST**...
1. Provide a user interface for chatbot interactions and optional file uploads.
2. Allow users to upscale images (PDF, JPEG, PNG, ~~BMP~~, ~~SVG~~) without limits.
3. Let users generate tasks based on their speech
4. Include a privacy policy that clearly outlines data usage when interacting with GPT-3.5 or Gemini-Pro

## "Non-Functional Requirements

The application **SHOULD**...
1. Aim for 24/7 availability.
2. Use a multipage Streamlit application structure.
3. Prioritize ease of navigation
4. Feature a visually appealing design.

# Roadmap

## Thinking Out Loud
I'm focusing on overhauling the retrieval prompting logic. I'll remove `ConversationalRetrievalChain` because it restricts flexibility and nuanced answers. I'll either adopt `AgentExecutor` or implement LangGraph for greater control over AI decision-making, hopefully improving response quality.

## Future Functionality Plans

- [x] Create an RAG chatbot
- [x] Add Gemini-Pro to the model list
- [x] Add Anthropic's Claude 3 model family
- [ ] Add AI decision making
  - [ ] Implement Corrective-RAG OR Reflective-RAG
- [x] Turn into a Multi-Page Application (MPA)
  - [x] (Homepage) Add a welcome screen with...
    - [x] a description of the project
    - [x] privacy policy
  - [x] (Page) Migrate RAG SPA code
    - [x] Add "Temperature" slider
  - [ ] (Page) Add a "Task Transcriber"
    - [ ] Microphone input (Record in browser)
    - [ ] Transcribes audio to text
    - [ ] Use LLM to identify each and every task while grouping some to avoid redundance
    - [ ] Generates text within a predefined task template, for each task identified
  - [x] (Page) Add "Image Upscaler"
    - [x] Multi-file upload
    - [x] File type detection
  - [ ] (Page) Add "Object Removal" from images
    - [x] Review HuggingFace Spaces's as a potential solution

---

# [License](./LICENSE)

# Privacy Policy
A conglomerate privacy policy governing FreeStream is planned. For the time being, please refer to the privacy policies of the underlying foundational AI model providers.

- [OpenAI Privacy Policy](https://openai.com/policies/privacy-policy)
- [Google](https://transparency.google/our-policies/privacy-policy-terms-of-service/ "Was unable to find a privacy policy specific to Google AI Studio.")
- [Anthropic](https://support.anthropic.com/en/articles/7996866-how-long-do-you-store-personal-data "Support forum response that may suddenly be obsoleted.")
- [Streamlit](https://streamlit.io/privacy-policy/)
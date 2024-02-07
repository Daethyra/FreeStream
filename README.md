# FreeStream

## Description
With this project I aim to build a reliable chatbot for my professional friends in law and medicine. I am to create a solution that is as privacy friendly as possible. However, currently the project only offers GPT-3.5-Turbo as it is the easiest and most consistent model to use in development testing. While OpenAI doesn't train models on data sent to the API, it is worth noting that this solution isn't finished considering there is no absolutely-privacy-friendly model installed, yet.

I wanted to build something that truly helps people, doesn't cost them any money, or require signing up for an account.

---

### Drawbacks:
- Currently only supports the GPT-3.5-Turbo model
- The implemented `qa_chain` forces answers to be based on the context, and the context only. This makes it difficult to interact with the chat history in a meaningful way

---

[License](./LICENSE)

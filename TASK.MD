# Data Scientist Home Assignment: Chatbot with RAG and OpenAI

## Overview
As part of the hiring process for the Data Scientist role, we’re asking candidates to complete a home assignment. The goal is to build a simple chat bot using RAG (Retrieval-Augmented Generation) and OpenAI’s language models.
This assignment is designed to evaluate your ability to:
- Understand and apply RAG principles
- Work with unstructured data
- Use vector data bases effectively
- Build a working prototype that demonstrates a product

## Assignment Description
Build a basic chat bot that answers questions based on a set of provided bank policy documents.
The chat bot should:
1. Index the provided documents into a vector store.
2. On user input, embed the question, search for relevant documents from the vector store, and
3. Send the context and question to OpenAI for answer generation.
4. Chat evaluation
    a. Method of evaluation.
    b. Testing with different emedders.
You may use a local vector DB (using Docker or local installation) or a cloud - based one - whatever works best for you.

## Deliverables
Please submit:
- A short README describing your solution, how to use it, setup instructions, and any assumptions you made.
- Source code in Python.
You can submit via GitHub link, or zip file.

## Time Expectation
We estimate this task should take 2–3 hours of focused work. If you encounter a time-consuming problem, please contact us.
Don’t overengineer, but keep things simple and readable.

## Notes on the Documents
The documents provided are general bank policy documents meant for internal orientation. They do not contain any sensitive or confidential information. Still, please do not share them beyond the scope of this assignment.

## What We’re Looking For
- Clear and simple architecture
- Proper use of retrieval + generation
- Understanding of embeddings and similarity search
- Clean and readable code
- Bonus: UX polish (CLI or Web - your choice)
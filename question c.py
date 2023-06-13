import openai
import numpy as np


openai.api_key = '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'


stored_vectors = np.array([
    [0.1, 0.2, 0.3, ...],  # Embedding for document 1
    [0.4, 0.5, 0.6, ...],  # Embedding for document 2
    [0.7, 0.8, 0.9, ...],  # Embedding for document 3
    ...
],dtype=object)
stored_documents = [
    "Document 1",
    "Document 2",
    "Document 3",
    ...
]


user_question = "What is the answer to my question?"


response = openai.Completion.create(
    engine="davinci",
    prompt=user_question,
    max_tokens=0,
    temperature=0,
    n=1,
    stop=None,
)
user_question_embedding = response.choices[0].embedding


document_scores = np.dot(stored_vectors, user_question_embedding) / (
    np.linalg.norm(stored_vectors, axis=1) * np.linalg.norm(user_question_embedding)
)
most_similar_document_index = np.argmax(document_scores)
relevant_document = stored_documents[most_similar_document_index]


response = openai.Completion.create(
    engine="davinci",
    prompt=relevant_document + "\nQuestion: " + user_question + "\nAnswer:",
    max_tokens=50,
    temperature=0.8,
    n=1,
    stop=None,
)


generated_answer = response.choices[0].text.strip()

print("Relevant Document:")
print(relevant_document)
print("Generated Answer:")
print(generated_answer)

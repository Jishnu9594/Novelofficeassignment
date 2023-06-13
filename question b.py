import openai


openai.api_key ='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'


original_document = '''
This is the original document. It contains multiple sentences that we want to split into separate documents. The goal is to generate embeddings for each individual document.
'''


sentences = original_document.split('. ')


split_documents = []
split_embeddings = []


for sentence in sentences:

    sentence = sentence.strip()


    if len(sentence) == 0:
        continue


    response = openai.Completion.create(
        engine="davinci",
        prompt=sentence,
        max_tokens=0,
        temperature=0,
        n=1,
        stop=None,
    )
    embedding = response.choices[0].embedding


    split_documents.append(sentence)
    split_embeddings.append(embedding)


for i in range(len(split_documents)):
    print(f"Document {i+1}:")
    print(split_documents[i])
    print("Embedding:")
    print(split_embeddings[i])
    print()


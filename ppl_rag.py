from datasets import load_dataset
from haystack import Document
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader
from haystack.components.embedders import SentenceTransformersTextEmbedder,SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from uuid import uuid4
import re


def read_markdown(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        data = f.read()

    pattern = re.compile(r"\*\*\[(.*?)\]\(.*?\)\*\*\s+(.*)")
    matches = pattern.findall(data)

    _documents = []
    for match in matches:
        section = match[0]  
        text = match[1].strip() 
        text = " ".join(text.splitlines())
        
        _documents.append({
            "meta": section,
            "content": text
        })

    # print(documents)
    return _documents


def prepare_dataset(file_name):
    _documents = read_markdown(file_name)

    documents = [
        Document(
            content=doc["content"],
            meta={"section": doc.get("meta", "")},  
            id=str(uuid4()) 
        )
        for doc in _documents
    ]

    return documents


def prepare_pipeline(documents):
    document_store = InMemoryDocumentStore()

    model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

    indexing_pipeline = Pipeline()

    indexing_pipeline.add_component(instance=SentenceTransformersDocumentEmbedder(model=model), name="embedder")
    indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
    indexing_pipeline.connect("embedder.documents", "writer.documents")

    indexing_pipeline.run({"documents": documents})
    

    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    reader = ExtractiveReader()
    reader.warm_up()

    extractive_qa_pipeline = Pipeline()

    extractive_qa_pipeline.add_component(instance=SentenceTransformersTextEmbedder(model=model), name="embedder")
    extractive_qa_pipeline.add_component(instance=retriever, name="retriever")
    extractive_qa_pipeline.add_component(instance=reader, name="reader")

    extractive_qa_pipeline.connect("embedder.embedding", "retriever.query_embedding")
    extractive_qa_pipeline.connect("retriever.documents", "reader.documents")

    return extractive_qa_pipeline



if __name__ == "__main__":
    documents = prepare_dataset('witt_test/tractatus.md')
    extractive_qa_pipeline = prepare_pipeline(documents)
    query = "When should one be silent?"
    res = extractive_qa_pipeline.run(
        data={"embedder": {"text": query}, "retriever": {"top_k": 3}, "reader": {"query": query, "top_k": 2}}
    )

    for i in res.get("reader")["answers"]:
        if i.data is None:
            break
        print(i.data)
        print(i.document.meta['section'])
        print(i.document.content)
        print(i.score)


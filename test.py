import ppl_rag
import csv

def read_queries_from_md(file_name):
    queries = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            query = line.strip()
            if query:  
                queries.append(query)
    return queries


def save_answers_to_csv(answers, file_name):
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Query","Answer", "Section", "Content", "Score"]) 

            for answer in answers:
                writer.writerow([answer.query, answer.data, answer.document.meta['section'], answer.document.content, answer.score])


if __name__ == "__main__":
    doc_name = 'tractatus.md'
    documents = ppl_rag.prepare_dataset(doc_name)
    extractive_qa_pipeline = ppl_rag.prepare_pipeline(documents)
    queries = read_queries_from_md('query_list.md')

    all_answers = []

    for query in queries:
        res = extractive_qa_pipeline.run(
            data={"embedder": {"text": query}, "retriever": {"top_k": 3}, "reader": {"query": query, "top_k": 2}}
        )

        for i in res.get("reader")["answers"]:
            if i.data is None:
                break
            all_answers.append(i)

    save_answers_to_csv(all_answers,'answers_list.csv')

import re
import json

"""match pattern: number/text"""

def dataset_prepare(markdown_content):
    pattern = re.compile(r"\*\*\[(.*?)\]\(.*?\)\*\*\s+(.*)")
    matches = pattern.findall(markdown_content)

    data = []
    for match in matches:
        reference_number, content = match
        conversation = {
            "conversations": [
                {"from": "human", "value": f"[{reference_number}]"},
                {"from": "gpt", "value": content.strip()}
            ]
        }
        data.append(conversation)
    return data


if __name__ == "__main__":
    with open("tractatus.md", "r", encoding="utf-8") as f:
        markdown_content = f.read()

    data = dataset_prepare(markdown_content)

    with open("tractatus.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

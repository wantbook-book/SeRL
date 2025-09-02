import json

def evaluate(input_file):
    corr = 0
    wrong = 0
    with open(input_file, "r") as fi:
        for line in fi:
            item = json.loads(line)
            if item["pred"] == item["answer_index"]:
                corr += 1
            else:
                wrong += 1

    accu = corr / (corr + wrong)
    return accu, corr, wrong

if __name__ == '__main__':
    input_file = '/angel/fwk/code/SeRL/evaluation/GPQA/eval_results/angel/fwk/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/outputs.jsonl'
    accu, corr, wrong = evaluate(input_file)
    output_file = input_file.replace('outputs.jsonl', 'metric.json')
    with open(output_file, 'w') as f:
        json.dump({"accu": accu, "corr": corr, "wrong": wrong}, f)
    print(f"Accuracy: {accu:.4f}, Correct: {corr}, Wrong: {wrong}")


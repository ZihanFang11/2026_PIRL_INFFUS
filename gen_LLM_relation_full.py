import os
import re
import json
import itertools
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM
import warnings
warnings.filterwarnings("ignore")
ALLOWED_RELATIONS = {
    "likely co-occurring",
    "mutually exclusive",
    "hierarchical / entailment",
    "mostly independent"
}


def load_llm(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


    if 'DeepSeek' in model_name:
        model = Qwen2ForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    model.eval()
    return tokenizer, model


def load_label_definitions(xlsx_path):
    df = pd.read_excel(xlsx_path)
    if "label" not in df.columns or "definition" not in df.columns:
        raise ValueError(f"{xlsx_path} must contain 'label' and 'definition' columns.")

    df["label"] = df["label"].astype(str).str.strip()
    df["definition"] = df["definition"].astype(str).str.strip()

    df = df[(df["label"] != "") & (df["definition"] != "")]
    return df[["label", "definition"]].drop_duplicates(subset=["label"], keep="last")


def build_relation_prompt(label1, label2,dataset):
    if dataset=="3sources":
        prompt = f"""
        You are a professional news topic annotation expert specializing in multi-label news classification.
        
        Given two news topic labels, determine their primary relationship with respect to whether they are likely to be assigned together to the same news story or news article.
        
        Focus on journalistic topic co-occurrence, topical compatibility, and common news annotation practice. Do not judge only by lexical similarity or abstract semantic relatedness; instead, prioritize whether the two labels would realistically co-occur in the same news report.
        
        The candidate labels come from a six-topic news dataset:
        business, entertainment, health, politics, sport, technology.
        
        Relation candidates:
        1. likely co-occurring
           - the two labels are often assigned together to the same news story because the story commonly involves both topics
        
        2. mutually exclusive
           - the two labels are unlikely to be assigned together to the same news story under normal news reporting conditions
        
        3. hierarchical / entailment
           - one label is a broader or narrower topical category of the other, so the presence of one may imply the other at the annotation level
        
        4. mostly independent
           - the two labels do not show a strong tendency to co-occur or exclude each other in the same news story


        Label 1: {label1}
        Label 2: {label2}

        Return format:
        {{
          "relation_type": "...",
          "confidence": 0.0,
          "explanation": "..."
        }}
        """
    elif dataset=="emotions":
        prompt = f"""
        You are a professional music emotion annotation expert specializing in multi-label emotion classification for music clips.

        Given two emotion labels, determine their primary relationship with respect to whether they are likely to be assigned together to the same music excerpt.

        Focus on music emotion co-occurrence, affective compatibility, and common multi-label annotation practice for musical pieces. Do not judge only by lexical similarity or abstract semantic relatedness; instead, prioritize whether the two labels would realistically co-occur in the same music clip based on its emotional expression.

        The candidate labels come from a six-label music emotion dataset:
        amazed-suprised, happy-pleased, relaxing-calm, quiet-still, sad-lonely, angry-aggresive.

        Relation candidates:
        1. likely co-occurring
           - the two labels are often assigned together to the same music clip because the clip commonly conveys both emotions at the same time

        2. mutually exclusive
           - the two labels are unlikely to be assigned together to the same music clip because their emotional expressions are usually incompatible

        3. hierarchical / entailment
           - one label is broader, narrower, or strongly implied by the other at the annotation level, so the presence of one may suggest the other

        4. mostly independent
           - the two labels do not show a strong tendency to co-occur or exclude each other in the same music clip

        Important guidelines:
        - Consider emotional compatibility in musical expression rather than general word meaning.
        - Consider that music can express blended or mixed emotions.
        - Use common sense about music mood annotation.
        - Prefer co-annotation likelihood over pure semantic similarity.
        - Treat the relation as directional if needed: the presence of Label 1 may affect the likelihood of Label 2 differently than the reverse.

        Label 1: {label1}
        Label 2: {label2}

        Return format:
        {{
          "relation_type": "...",
          "confidence": 0.0,
          "explanation": "..."
        }}
        """
    elif dataset == "mirflickr":
        prompt = f"""
You are an expert in multi-label image annotation for the MIR-Flickr dataset.
Some labels may appear in two forms, such as "car" and "car*".
In this dataset, the label with "*" denotes a stricter / narrower annotation version of the same concept,
while the label without "*" denotes a broader annotation version.
Therefore, treat them as related but not always equivalent.
If one label is a stricter version of the other, prefer "hierarchical / entailment".

Your task is to determine the primary relationship between two labels
from the perspective of image-level co-annotation.

Relation candidates:
1. likely co-occurring
   - the two labels often appear together in the same image, scene, or visual context

2. mutually exclusive
   - the two labels are unlikely to be assigned to the same image under normal visual conditions

3. hierarchical / entailment
   - one label is a subtype, parent type, or a more specific visual instance of the other, so the presence of one may imply the other at the annotation level

4. mostly independent
   - the two labels have no strong tendency to appear together or exclude each other in the same image

Choose the single most appropriate relation based on image annotation semantics.
Instructions:
- Choose exactly one primary relation.
- Return a confidence score between 0 and 1.
- Provide a short explanation in one or two sentences.
- Output MUST be valid JSON only.
- Use exactly these keys: relation_type, confidence, explanation


Label 1: {label1}
Label 2: {label2}

Return format:
{{
  "relation_type": "...",
  "confidence": 0.0,
  "explanation": "..."
}}
"""

    else:
        prompt = f"""
    You are a professional image annotation expert specializing in multi-label image datasets.
Given two image labels, determine their primary relationship with respect to whether they are likely to appear in the same image annotation.
Focus on visual co-occurrence, scene compatibility, object compatibility, and image-level annotation practice, rather than purely linguistic or abstract semantic relations.

Relation candidates:
1. likely co-occurring
   - the two labels often appear together in the same image, scene, or visual context

2. mutually exclusive
   - the two labels are unlikely to be assigned to the same image under normal visual conditions

3. hierarchical / entailment
   - one label is a subtype, parent type, or a more specific visual instance of the other, so the presence of one may imply the other at the annotation level

4. mostly independent
   - the two labels have no strong tendency to appear together or exclude each other in the same image

Choose the single most appropriate relation based on image annotation semantics.
Instructions:
- Choose exactly one primary relation.
- Return a confidence score between 0 and 1.
- Provide a short explanation in one or two sentences.
- Output MUST be valid JSON only.
- Use exactly these keys: relation_type, confidence, explanation

Label 1: {label1}
Label 2: {label2}

Return format:
{{
  "relation_type": "...",
  "confidence": 0.0,
  "explanation": "..."
}}
"""
    return prompt.strip()


def extract_json(text):
    text = text.strip()

    # 去掉可能的 ```json ``` 包裹
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # 直接尝试整体解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 提取第一个 {...}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        return json.loads(candidate)

    raise ValueError(f"Cannot parse JSON from response: {text}")


def normalize_relation(rel):
    rel = str(rel).strip().lower()
    rel = re.sub(r"\s*/\s*", " / ", rel)  # normalize slash spacing

    mapping = {
        "co-occurring": "likely co-occurring",
        "cooccurring": "likely co-occurring",
        "likely cooccuring": "likely co-occurring",
        "likely co-occurring": "likely co-occurring",
        "likely co-occurrence": "likely co-occurring",
        "co-occurrence": "likely co-occurring",

        "mutually exclusive": "mutually exclusive",
        "exclusive": "mutually exclusive",

        "hierarchical": "hierarchical / entailment",
        "entailment": "hierarchical / entailment",
        "hierarchical / entailment": "hierarchical / entailment",
        "hierarchical/entailment": "hierarchical / entailment",
        "hierarchy": "hierarchical / entailment",

        "independent": "mostly independent",
        "mostly independent": "mostly independent"
    }

    return mapping.get(rel, rel)

@torch.no_grad()
def infer_relation(dataset,tokenizer, model, label1, label2,  max_new_tokens=520):
    prompt = build_relation_prompt(label1, label2,dataset)

    messages = [
        {
            "role": "system",
            "content": (
                "You must return valid JSON only."
            )
        },
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)


    if 'DeepSeek' in model_name:
        max_new_tokens=1024
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.3,
            top_p=0.8,

        )
    else:

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,

        )
    output_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(response)
    data = extract_json(response)

    relation_type = normalize_relation(data.get("relation_type", ""))
    confidence = data.get("confidence", 0.0)
    explanation = str(data.get("explanation", "")).strip()

    if relation_type not in ALLOWED_RELATIONS:
        raise ValueError(f"Invalid relation_type: {relation_type}")

    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    return {
        "relation_type": relation_type,
        "confidence": confidence,
        "explanation": explanation
    }


def make_pair_key(label1, label2):
    a, b = sorted([str(label1).strip(), str(label2).strip()])
    return f"{a}|||{b}"


def make_pair_key(label_1, label_2):
    # 有向 key，A->B 与 B->A 不同
    return f"{label_1}|||{label_2}"


def gen_label_relations(dataset, label_list, output_path, model_name):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    items = label_list

    # 构造所有有序标签对：(A,B) 和 (B,A) 都保留，去掉 (A,A)
    pairs = []
    for x in items:
        for y in items:
            if x != y:
                pairs.append((x, y))

    print(f"Total labels: {len(items)}")
    print(f"Total ordered pairs: {len(pairs)}")

    # 断点续跑
    if os.path.exists(output_path):
        old_df = pd.read_excel(output_path)

        required_cols = {
            "label_1", "label_2", "relation_type", "confidence", "explanation"
        }
        missing = required_cols - set(old_df.columns)
        if missing:
            raise ValueError(f"Existing file missing columns: {missing}")

        old_df["pair_key"] = old_df.apply(
            lambda r: make_pair_key(r["label_1"], r["label_2"]), axis=1
        )

        # 只把成功结果视为完成
        done_df = old_df[
            old_df["relation_type"].notna() &
            old_df["confidence"].notna() &
            old_df["explanation"].notna() &
            (old_df["relation_type"] != "[ERROR]")
        ].copy()

        done_keys = set(done_df["pair_key"].tolist())
        results = old_df.drop(columns=["pair_key"]).to_dict("records")

        print(f"Loaded existing results: {len(done_keys)} finished ordered pairs")
    else:
        done_keys = set()
        results = []

    remain_pairs = [
        (l1, l2)
        for (l1, l2) in pairs
        if make_pair_key(l1, l2) not in done_keys
    ]

    print(f"Remaining ordered pairs: {len(remain_pairs)}")

    if len(remain_pairs) == 0:
        print("All ordered pairs already processed.")
        return

    tokenizer, model = load_llm(model_name)

    total = len(remain_pairs)
    for i, (label1, label2) in enumerate(remain_pairs, 1):
        try:
            pred = infer_relation(dataset, tokenizer, model, label1, label2)
            print(pred)
            row = {
                "label_1": label1,
                "label_2": label2,
                "relation_type": pred["relation_type"],
                "confidence": pred["confidence"],
                "explanation": pred["explanation"]
            }
        except Exception as e:
            row = {
                "label_1": label1,
                "label_2": label2,
                "relation_type": "[ERROR]",
                "confidence": 0.0,
                "explanation": str(e)
            }

        print(
            f"[{i}/{total}] {label1} -> {label2} "
            f"=> {row['relation_type']} ({row['confidence']})"
        )

        results.append(row)

        out_df = pd.DataFrame(results)
        out_df["pair_key"] = out_df.apply(
            lambda r: make_pair_key(r["label_1"], r["label_2"]), axis=1
        )
        out_df = out_df.drop_duplicates(subset=["pair_key"], keep="last")
        out_df = out_df.drop(columns=["pair_key"])
        out_df.to_excel(output_path, index=False)

    print(f"Saved to: {output_path}")
def load_labels(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        labels = [line.rstrip('\n') for line in f]
    return labels

if __name__ == "__main__":
    path = "./pretrain_models"
    llms = ["Llama-3.1-8B-Instruct",'Qwen3-30B-A3B-Instruct-2507',"DeepSeek-R1-Distill-Qwen-14B"]
    llms = ["Qwen-2.5-14B-Instruct"]


    datas = ["emotions", '3sources', "VOC07","mirflickr", "corel5k","IAPRTC12"]

    for dataset in datas:
        for llm in llms:
            output_path = f"labels_relation_llm/{dataset}_{llm}_relations.xlsx"
            model_name = f"{path}/{llm}"

            label_list = load_labels(f'./data/label/{dataset}.txt')
            print(f"Dataset: {dataset}, LLM: {llm}")
            gen_label_relations(dataset,label_list,output_path, model_name)
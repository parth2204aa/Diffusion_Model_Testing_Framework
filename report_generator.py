### report_generator.py
import pandas as pd
from metrics.clip_score import clip_similarity

def generate_report(results):
    records = []
    for prompt, img_path in results:
        sim = clip_similarity(img_path, prompt)
        records.append({
            "Prompt": prompt,
            "Image Path": img_path,
            "CLIP Similarity": sim
        })
    df = pd.DataFrame(records)
    df.to_csv("qa_report.csv", index=False)
    print(df)

import os
import json
import xml.etree.ElementTree as ET

MEDQUAD_DIR = "MedQuAD"  # path to cloned repo folder
TARGET_SUBFOLDER = "4_MPlus_Health_Topics_QA"
OUTPUT_FILE = "mplus_health_topics_knowledge_base.json"

def parse_mplus_health_topics():
    all_qa_pairs = []
    skipped = 0

    folder_path = os.path.join(MEDQUAD_DIR, TARGET_SUBFOLDER)

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"[ERROR] Subfolder not found: {folder_path}")

    for filename in os.listdir(folder_path):
        if not filename.endswith(".xml"):
            continue
        filepath = os.path.join(folder_path, filename)

        try:
            tree = ET.parse(filepath)
            root = tree.getroot()

            focus = root.findtext("Focus", default="").strip()
            source = root.attrib.get("source", "MedlinePlus Health Topics")

            for qa_pair in root.iter("QAPair"):
                question = qa_pair.findtext("Question", default="").strip()
                answer = qa_pair.findtext("Answer", default="").strip()

                # Skip empty or very short entries
                if not question or not answer or len(answer) < 20:
                    continue

                all_qa_pairs.append({
                    "condition": focus,
                    "question": question,
                    "content": answer,
                    "source": source
                })
        except ET.ParseError:
            skipped += 1
            continue

    print(f"[INFO] Parsed {len(all_qa_pairs)} QA pairs from '{TARGET_SUBFOLDER}'. Skipped {skipped} malformed files.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_qa_pairs, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    parse_mplus_health_topics()
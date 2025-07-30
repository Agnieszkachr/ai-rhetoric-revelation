# src/analysis_utils.py

import pandas as pd
import numpy as np
import itertools
import re
from scipy.spatial.distance import cosine
#for 01
import json
import os
#import re # already above
from collections import OrderedDict
import numpy as np
#end for 01

def build_runs_df(raw_data):
    """Parses the raw LLM output JSON into a tidy DataFrame of all runs."""
    rows = []
    if isinstance(raw_data, dict) and "narrative_units" in raw_data:
        units = raw_data["narrative_units"]
    elif isinstance(raw_data, list):
        units = raw_data
    else:
        return pd.DataFrame()

    for unit in units:
        if "validation_stats" in unit:
            uid = unit["unit_id"]
            verse_range = unit.get("verse_range", "N/A")
            stats = unit["validation_stats"]
            n_iter = len(next(iter(stats.values()))["results"])
            for i in range(n_iter):
                vec = {cat: stats[cat]["results"][i] for cat in stats}
                rows.append({"unit": uid, "run": f"r{i:02d}", "verse_range": verse_range, **vec})
    return pd.DataFrame(rows)

def normalise_uid(u):
    """Standardizes unit IDs to the format 'unit_001'."""
    m = re.search(r'\\d+', str(u))
    return f"unit_{int(m.group()):03d}" if m else str(u).strip()

def mean_pairwise_cos(df, category_list):
    """Calculates the mean pair-wise cosine similarity for intra-model stability."""
    vals = []
    for _, sub in df.groupby("unit"):
        vecs = sub[category_list].to_numpy()
        if len(vecs) < 2: continue
        vals.append(np.mean([1 - cosine(a, b) for a, b in itertools.combinations(vecs, 2)]))
    return np.mean(vals) if vals else np.nan
    
def parse_data_to_df(json_data):
    """
    Parses the final aggregated LLM output JSON into a clean DataFrame.
    Extracts the final mean rhetorical scores for each narrative unit.
    """
    records = []
    # Check if 'narrative_units' exists, otherwise assume the data is the list of units
    if isinstance(json_data, dict) and 'narrative_units' in json_data:
        units = json_data['narrative_units']
    elif isinstance(json_data, list):
        units = json_data
    else:
        raise ValueError("JSON data format not recognized (missing 'narrative_units' or not a list).")

    for unit in units:
        if 'final_rhetorical_vector' in unit:
            record = {
                'unit_id': unit['unit_id'],
                'verse_range': unit.get('verse_range', 'N/A') # Use .get for safety
            }
            record.update(unit['final_rhetorical_vector'])
            records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.set_index('unit_id', inplace=True)
    return df

# for 01
def load_json_file(file_path):
    """Loads a JSON file with robust error handling."""
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Use OrderedDict to preserve key order from the JSON file
            return json.load(f, object_pairs_hook=OrderedDict)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to decode JSON from {file_path}. Details: {e}")
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while reading {file_path}. Details: {e}")
        return None

def clean_json_response(text):
    """Strip markdown fences etc. and return the substring that looks like JSON."""
    start = text.find("{")
    end   = text.rfind("}")
    return None if start == -1 or end == -1 else text[start : end + 1]
    
def _balance_brackets(text: str) -> str:
    """
    If the AI response is cut off mid-stream, we may have more opening than
    closing braces/brackets.  This function appends the missing closers in
    LIFO order so that the string becomes syntactically valid JSON.
    """
    stack = []
    for ch in text:
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and ch == stack[-1]:
            stack.pop()

    # append any missing closers
    text += "".join(reversed(stack))
    return text

def safe_json_loads(text):
    try:
        return json.loads(text, object_pairs_hook=OrderedDict)
    except json.JSONDecodeError:
        json5 = ensure_package("json5")           # installs on-demand
        try:
            return json5.loads(text, object_pairs_hook=OrderedDict)
        except Exception:
            # Last resort: balance braces/brackets, then try json5 again
            fixed = _balance_brackets(text)
            return json5.loads(fixed, object_pairs_hook=OrderedDict)

def _extract_justification(raw_text: str) -> str | None:
    """
    Robustly pull a justification paragraph from raw model output.

    1.  JSON key/value pattern  →  "justification": "...."
    2.  Section heading pattern →  Justification: ....
    """
    import re

    # ------------------------------------------------------------
    # 1) Try strict JSON-style key/value first
    # ------------------------------------------------------------
    m = re.search(
        r'"?justification"?\s*:\s*'
        r'("([^"\\]|\\.)*"|\'([^\'\\]|\\.)*\'|`([^`\\]|\\.)*`)',  # any quoted string
        raw_text,
        flags=re.I | re.S,
    )
    if m:
        txt = m.group(1).strip("`'\"")       # strip surrounding quotes / back-ticks
        return txt.strip()

    # ------------------------------------------------------------
    # 2) Fallback: look for a heading then grab following lines
    # ------------------------------------------------------------
    lines = raw_text.splitlines()
    capture = False
    buf = []
    for line in lines:
        if capture:
            # stop at blank line, code fence, or new JSON/object start
            if (
                not line.strip()                         # empty line
                or re.match(r"\s*```", line)             # code fence
                or re.match(r"\s*[{\[]\s*$", line)       # opening brace on its own
                or re.match(r"\s*\w+\s*[:=]\s*[{[\"'\d]", line)  # new key/value
            ):
                break
            buf.append(line.strip())
        else:
            if re.match(r"\s*(justification|explanation|reasoning)\s*[:\-]?\s*$",
                        line, flags=re.I):
                capture = True

    return " ".join(buf).strip() or None

def _clean_justification(txt: str | None) -> str | None:
    """Normalise quotes, collapse whitespace, strip code fences."""
    if not txt:
        return None
    import re
    txt = txt.replace("“", '"').replace("”", '"')         # curly → straight
    txt = txt.replace("’", "'").replace("‘", "'")
    txt = re.sub(r"`{3}.*?`{3}", "", txt, flags=re.S)     # remove ``` blocks
    txt = re.sub(r"\s+", " ", txt)                        # collapse whitespace
    return txt.strip()

def validate_vector(vector, concepts):
    """
    Ensure vector has numeric values for every concept & sums to 100.
    Returns (possibly-normalised vector, normalized_flag, message)
    """
    if not vector or not isinstance(vector, dict):
        return vector, False, "Rhetorical vector is missing or not a dictionary."

    for c in concepts:
        if c not in vector or not isinstance(vector[c], (int, float)):
            return vector, False, f"Vector missing concept '{c}' or value is not numeric."

    total = sum(vector.values())
    if np.isclose(total, 100.0, atol=0.01):          # Already OK
        return vector, False, "Vector sum is valid."

    # Normalise
    norm = OrderedDict()
    if total != 0:
        for k, v in vector.items():
            norm[k] = round(v / total * 100.0, 2)
        norm_total = sum(norm.values())
        if not np.isclose(norm_total, 100.0, atol=0.01):
            biggest = max(norm, key=norm.get)
            norm[biggest] += round(100.0 - norm_total, 2)

    return norm, True, f"Vector sum was {total:.2f}, normalised to 100."















# ADD THESE FUNCTIONS TO THE END of src/analysis_utils.py

def clean_json_response(text):
    """Strip markdown fences etc. and return the substring that looks like JSON."""
    start = text.find("{")
    end   = text.rfind("}")
    return None if start == -1 or end == -1 else text[start : end + 1]

def safe_json_loads(text):
    """Tolerantly load a JSON string, trying json5 and bracket balancing as fallbacks."""
    try:
        return json.loads(text, object_pairs_hook=OrderedDict)
    except json.JSONDecodeError:
        import json5
        try:
            return json5.loads(text, object_pairs_hook=OrderedDict)
        except Exception:
            fixed = _balance_brackets(text)
            return json5.loads(fixed, object_pairs_hook=OrderedDict)

def _balance_brackets(text: str) -> str:
    """Appends missing closing brackets/braces to a potentially truncated JSON string."""
    stack = []
    for ch in text:
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and ch == stack[-1]:
            stack.pop()
    text += "".join(reversed(stack))
    return text

def _extract_justification(raw_text: str) -> str | None:
    """Robustly pull a justification paragraph from raw model output."""
    # ... (copy the full function from your notebook)
    m = re.search(r'"?justification"?\s*:\s*("([^"\\]|\\.)*"|\'([^\'\\]|\\.)*\'|`([^`\\]|\\.)*`)', raw_text, flags=re.I | re.S)
    if m:
        return m.group(1).strip("`'\"").strip()
    lines, capture, buf = raw_text.splitlines(), False, []
    for line in lines:
        if capture:
            if not line.strip() or re.match(r"\s*```", line) or re.match(r"\s*[{\[]\s*$", line) or re.match(r"\s*\w+\s*[:=]\s*[{[\"'\d]", line):
                break
            buf.append(line.strip())
        elif re.match(r"\s*(justification|explanation|reasoning)\s*[:-]?\s*$", line, flags=re.I):
            capture = True
    return " ".join(buf).strip() or None

def _clean_justification(txt: str | None) -> str | None:
    """Normalise quotes, collapse whitespace, strip code fences."""
    # ... (copy this entire function from your notebook)
    if not txt: return None
    txt = txt.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    txt = re.sub(r"`{3}.*?`{3}", "", txt, flags=re.S)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def validate_vector(vector, concepts):
    """Ensure vector has numeric values for every concept & sums to 100."""
    # ... (copy the full function from your notebook)
    if not vector or not isinstance(vector, dict):
        return vector, False, "Rhetorical vector is missing or not a dictionary."
    for c in concepts:
        if c not in vector or not isinstance(vector[c], (int, float)):
            return vector, False, f"Vector missing concept '{c}' or value is not numeric."
    total = sum(vector.values())
    if np.isclose(total, 100.0, atol=0.01):
        return vector, False, "Vector sum is valid."
    norm = OrderedDict()
    if total != 0:
        for k, v in vector.items():
            norm[k] = round(v / total * 100.0, 2)
        norm_total = sum(norm.values())
        if not np.isclose(norm_total, 100.0, atol=0.01):
            biggest = max(norm, key=norm.get)
            norm[biggest] += round(100.0 - norm_total, 2)
    return norm, True, f"Vector sum was {total:.2f}, normalised to 100."
#for 02
# In file: src/analysis_utils.py
# Add this new function to the end of the file.

import numpy as np
import pandas as pd

def interpolate_to_verses(segmentation_df: pd.DataFrame, score_df: pd.DataFrame, total_verses: int, num_categories: int) -> np.ndarray:
    """Interpolates rhetorical scores from narrative units to a per-verse array."""
    verse_scores = np.zeros((total_verses, num_categories))
    verse_idx = 0
    
    # Ensure unit_id is the index for quick lookup
    if 'unit_id' in score_df.columns:
        score_df = score_df.set_index('unit_id')

    for _, unit in segmentation_df.iterrows():
        num_vers_in_unit = unit['num_verses']
        mu_vector = score_df.loc[unit['unit_id']]['mu']
        
        end_idx = verse_idx + num_vers_in_unit
        if end_idx > total_verses:
            end_idx = total_verses
        
        for v_i in range(verse_idx, end_idx):
            verse_scores[v_i] = mu_vector
        
        verse_idx = end_idx
        
    if verse_idx != total_verses:
        print(f"Warning: Interpolated verses ({verse_idx}) do not match total ({total_verses}). Check segmentation.")
        
    return verse_scores
#end for 02    

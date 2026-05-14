import os
import sys
import json
import uuid
import shutil
import argparse
import subprocess
from datetime import datetime
import copy

MODELS = {
    'gan': 'GAN',
    'i2sb': 'I2SB',
    'rddm': 'RDDM'
}

def get_user_text(data):
    text = ""
    if isinstance(data, dict):
        for k, v in data.items():
            if k.lower() in ['question', 'prompt', 'text', 'message', 'content', 'user_input', 'query']:
                if isinstance(v, str):
                    text += v + " "
            text += get_user_text(v)
    elif isinstance(data, list):
        for item in data:
            text += get_user_text(item)
    return text

def extract_png_paths(data):
    paths = []
    if isinstance(data, dict):
        if "source_file_path" in data and isinstance(data["source_file_path"], str) and os.path.exists(data["source_file_path"]):
            paths.append(data["source_file_path"])
        if "source_path" in data and isinstance(data["source_path"], str) and os.path.exists(data["source_path"]):
            paths.append(data["source_path"])
        for k, v in data.items():
            paths.extend(extract_png_paths(v))
    elif isinstance(data, list):
        for v in data:
            paths.extend(extract_png_paths(v))
    elif isinstance(data, str):
        if (data.lower().endswith('.png') or data.lower().endswith('.jpg') or data.lower().endswith('.jpeg')) and os.path.exists(data):
            paths.append(data)
    return list(set(paths))

def get_b64_img(stdout_txt, model_dir):
    try:
        import re
        match = re.search(r"saved (?:in|at|to):?\s*['\"]?(.*?)['\"]?$", stdout_txt, re.MULTILINE|re.IGNORECASE)
        if match:
            out_dir = match.group(1).strip()
            out_dir = os.path.normpath(os.path.join(model_dir, out_dir))
            return f"\n- **Images Saved At**: `{out_dir}`"
    except Exception:
        pass
    return ""

def main():
    parser = argparse.ArgumentParser(description="ChatClinic LCR-MPI Tool Plugin")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input logic/JSON from ChatClinic")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output JSON for ChatClinic")
    
    args = parser.parse_args()
    
    with open(args.input, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    user_prompt = get_user_text(input_data).lower()
    
    print(f"Extracted User Prompt: '{user_prompt}'", file=sys.stderr)

    if 'gan' in user_prompt:
        model_arg = 'gan'
    elif 'i2sb' in user_prompt:
        model_arg = 'i2sb'
    elif 'rddm' in user_prompt:
        model_arg = 'rddm'
    else:
        model_arg = 'all'
        
    print(f"Final resolved model: '{model_arg}'", file=sys.stderr)
    
    models_to_run = list(MODELS.keys()) if model_arg == 'all' else [model_arg]

    for m in models_to_run:
        if m not in MODELS:
            print(f"Error: Unknown model '{m}' requested.", file=sys.stderr)
            sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    source_pngs = extract_png_paths(input_data)
    use_staging = len(source_pngs) > 0
    staging_input, staging_cond = None, None
    
    if use_staging:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
        staging_base = os.path.join(base_dir, 'staging', run_id)
        staging_input = os.path.join(staging_base, 'input')
        staging_cond = os.path.join(staging_base, 'input_cond')
        os.makedirs(staging_input, exist_ok=True)
        os.makedirs(staging_cond, exist_ok=True)
        
        for path in source_pngs:
            fname = os.path.basename(path)
            # Nomenclature rule: map condition file names back to input equivalents so I2SB finds them
            if '_cond' in fname.lower():
                new_fname = fname.lower().replace('_cond', '')
                shutil.copy2(path, os.path.join(staging_cond, new_fname))
            elif 'cond_' in fname.lower():
                new_fname = fname.lower().replace('cond_', '')
                shutil.copy2(path, os.path.join(staging_cond, new_fname))
            else:
                shutil.copy2(path, os.path.join(staging_input, fname))

    results = {}
    success_count = 0
    
    for model_name in models_to_run:
        model_dir = os.path.join(base_dir, MODELS[model_name])
        script_path = os.path.join(model_dir, 'inference.py')
        
        target_input = staging_input if use_staging else os.path.join(model_dir, 'input')
        
        cmd = ["python3", script_path, "--input", target_input]
        if model_name == 'i2sb':
            target_cond = staging_cond if use_staging else os.path.join(model_dir, 'input_cond')
            cmd.extend(["--cond", target_cond])
            
        print(f"Executing: {' '.join(cmd)}")
        
        env = copy.deepcopy(os.environ)
        
        try:
            completed = subprocess.run(
                cmd,
                cwd=model_dir,
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
                env=env
            )
            
            if completed.returncode == 0:
                success_count += 1
                img_md = get_b64_img(completed.stdout, model_dir)
                results[model_name] = {
                    "success": True,
                    "message": "Success",
                    "image": img_md
                }
            else:
                results[model_name] = {
                    "success": False,
                    "message": f"Inference failed with exit code {completed.returncode}. Stderr: {completed.stderr}"
                }
        except subprocess.TimeoutExpired:
            results[model_name] = {
                "success": False,
                "message": "Inference timed out after 1800 seconds."
            }
        except Exception as e:
            results[model_name] = {
                "success": False,
                "message": str(e)
            }

    output_data = {
        "success_count": success_count,
        "total": len(models_to_run),
        "summary": f"Inference completed! Successfully executed {success_count} models out of {len(models_to_run)}." + "".join([res.get("image", "") for res in results.values() if "image" in res]),
        "artifacts": {
            "inference_results": results
        },
        "status": "success" if success_count == len(models_to_run) else "partial_failure" if success_count > 0 else "failure",
        "provenance": {
            "tool_version": "1.0.0",
            "models_attempted": models_to_run
        }
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()

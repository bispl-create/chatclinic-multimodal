import os
import sys
import subprocess

# Define the models and their respective directories
MODELS = {
    "gan": "GAN",
    "i2sb": "I2SB",
    "rddm": "RDDM"
}

def run_inference(model_name, model_dir, seed=None):
    """Runs the inference.py script inside the specific model directory."""
    print(f"\n{'='*50}")
    print(f"Starting inference for: {model_name.upper()}")
    print(f"{'='*50}")

    # Check if the model directory and inference script exist
    inference_script = "inference.py"
    script_path = os.path.join(model_dir, inference_script)

    if not os.path.exists(script_path):
        error_msg = f"Inference script not found at {script_path}"
        print(f"Error: {error_msg}")
        return False, error_msg

    # Build the command
    cmd = [sys.executable, inference_script]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    try:
        # Run the command with cwd set to the model directory
        result = subprocess.run(cmd, cwd=model_dir, check=True, capture_output=True, text=True)
        print(f"\nSuccessfully completed inference for {model_name.upper()}")
        print("Standard Output:\n", result.stdout)
        return True, "Success"
    except subprocess.CalledProcessError as e:
        error_msg = f"Inference failed with exit code {e.returncode}. Stderr: {e.stderr}"
        print(f"\n{error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"\nUnexpected error running {model_name.upper()}: {e}")
        return False, error_msg

def run(payload: dict) -> dict:
    """ChatClinic entrypoint to run inference directly with a payload dict."""
    execution_context = payload.get("execution_context") or {}
    model_arg = payload.get('model', 'all')
    seed = payload.get('seed', None)

    # Determine which models to run
    models_to_run = list(MODELS.keys()) if model_arg == 'all' else [model_arg]

    # model_root: optional field to override where GAN/I2SB/RDDM folders are located.
    root_dir = payload.get('model_root') or os.path.dirname(os.path.abspath(__file__))

    results = {}
    success_count = 0

    for model_name in models_to_run:
        if model_name not in MODELS:
            results[model_name] = {"success": False, "message": f"Unknown model {model_name}"}
            continue

        model_dir = os.path.join(root_dir, MODELS[model_name])

        success, message = run_inference(model_name, model_dir, seed)
        if success:
            success_count += 1

        results[model_name] = {
            "success": success,
            "message": message
        }

    print(f"\n{'='*50}")
    print(f"All Tasks Finished! ({success_count}/{len(models_to_run)} successful)")
    print(f"{'='*50}\n")

    return {
        "summary": f"Inference completed! Successfully executed {success_count} models out of {len(models_to_run)}.",
        "artifacts": {
            "inference_results": results,
            "status": "success" if success_count == len(models_to_run) else "partial_failure",
            "execution_context": execution_context
        },
        "provenance": {
            "used_tools": ["lcr_mpi_tool"],
            "tool_version": "1.1.0"
        }
    }

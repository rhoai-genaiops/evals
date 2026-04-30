"""
Kubeflow Pipeline for Automated Canopy Testing with MLflow

This pipeline implements an automated testing workflow that:
1. Clones a git repository
2. Scans directories for test configuration files
3. Calls the backend to generate responses
4. Evaluates responses using MLflow scorers
5. Uploads an HTML summary to S3
"""

import kfp
from typing import NamedTuple, List
from kfp import dsl
from kfp.dsl import component
from kfp import kubernetes


@component(base_image='python:3.9')
def git_clone_op(
    repo_url: str,
    branch: str = "main"
):
    """Clone a Git repository into the shared PVC."""
    import os
    import subprocess
    import shutil
    from urllib.parse import urlparse, urlunparse

    folder = "/prompts"

    for entry in os.listdir(folder):
        path = os.path.join(folder, entry)
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    username = os.getenv("GIT_USERNAME")
    password = os.getenv("GIT_PASSWORD")

    if username and password:
        parsed = urlparse(repo_url)
        netloc = f"{username}:{password}@{parsed.hostname}"
        if parsed.port:
            netloc += f":{parsed.port}"
        repo_url = urlunparse(parsed._replace(netloc=netloc))

    print(f"Cloning {repo_url} at branch {branch} into {folder}")
    subprocess.run([
        "git", "clone",
        "--branch", branch,
        "--single-branch",
        "--depth", "1",
        repo_url,
        "/prompts"
    ], check=True)

    for item in os.listdir("/prompts"):
        print(item)


@component(base_image="python:3.9")
def scan_directory_op() -> NamedTuple("Output", [("configs", List[dict])]):
    """Scan /prompts for *_tests.yaml files."""
    import glob
    import os
    from collections import namedtuple

    configs = []
    base = "/prompts"

    for path in glob.glob(os.path.join(base, "**/**_tests.yaml"), recursive=True):
        rel_path = os.path.relpath(path, base)
        configs.append({"config_path": rel_path})

    print(f"Found {len(configs)} test config(s): {[c['config_path'] for c in configs]}")

    Output = namedtuple("Output", ["configs"])
    return Output(configs=configs)


@component(
    base_image="python:3.12",
    packages_to_install=["mlflow>=3.4.0", "httpx", "kubernetes"]
)
def run_all_mlflow_tests(
    configs: List[dict],
    backend_url: str,
    llm_endpoint: str,
    mlflow_tracking_uri: str,
    git_hash: str = "test",
):
    """Call the backend, then evaluate responses with MLflow scorers."""
    import os
    import json
    import yaml
    import mlflow
    from typing import Literal
    from mlflow.genai.judges import make_judge
    from mlflow.genai.scorers import scorer
    from urllib.parse import urljoin

    # MLflow setup
    os.environ["MLFLOW_TRACKING_AUTH"] = "kubernetes"
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
    os.environ["OPENAI_API_KEY"] = "no-key-required"

    namespace_path = "/run/secrets/kubernetes.io/serviceaccount/namespace"
    if os.path.exists(namespace_path):
        with open(namespace_path) as f:
            os.environ["MLFLOW_WORKSPACE"] = f.read().strip()

    token_path = "/run/secrets/kubernetes.io/serviceaccount/token"
    if os.path.exists(token_path):
        with open(token_path) as f:
            os.environ["MLFLOW_TRACKING_TOKEN"] = f.read().strip()

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    @scorer
    def is_shorter(outputs: str, inputs: dict) -> bool:
        """Is the response shorter than the input prompt?"""
        return len(outputs) < len(inputs.get("prompt", ""))

    # Backend helpers
    def send_request(payload, url):
        import httpx
        full_response = ""
        with httpx.Client(timeout=None) as http_client:
            with http_client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    error_body = response.read().decode()
                    raise RuntimeError(f"Backend returned {response.status_code}: {error_body}")
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[len("data: "):])
                            full_response += data.get("delta", "")
                        except json.JSONDecodeError:
                            continue
        return full_response

    def prompt_backend(prompt, endpoint):
        url = urljoin(backend_url, endpoint)
        return send_request({"prompt": prompt}, url)

    # Main loop
    repo_dir = "/prompts"

    for config_dict in configs:
        config_path = config_dict["config_path"]
        full_config_path = os.path.join(repo_dir, config_path)

        with open(full_config_path) as f:
            config = yaml.safe_load(f)

        usecase = config.get("usecase", config["name"])
        mlflow.set_experiment(usecase)

        endpoint = config["endpoint"]
        scorer_names = config.get("scorers", ["summary_quality", "is_shorter"])

        # Load judge prompt from file if specified in config
        judge_prompt_file = config.get("judge_prompt")
        if judge_prompt_file:
            judge_prompt_path = os.path.join(os.path.dirname(full_config_path), judge_prompt_file)
            with open(judge_prompt_path) as f:
                judge_instructions = f.read()
        else:
            judge_instructions = (
                "{{ inputs }}\n{{ outputs }}\n{{ expectations }}\n"
                "Is the response accurate and consistent with the expected response? "
                "Respond with only \"yes\" or \"no\"."
            )

        summary_quality_judge = make_judge(
            name="summary_quality",
            instructions=judge_instructions,
            feedback_value_type=Literal["yes", "no"],
            model="openai:/llama32",
            base_url=llm_endpoint + "/v1/chat/completions",
            extra_headers={"Authorization": "Bearer no-key-required"},
        )

        SCORER_MAP = {
            "summary_quality": summary_quality_judge,
            "is_shorter": is_shorter,
        }

        active_scorers = [SCORER_MAP[n] for n in scorer_names if n in SCORER_MAP]

        if not active_scorers:
            print(f"Warning: no recognised scorers in {config_path}, skipping.")
            continue

        # Generate responses from backend
        eval_data = []
        for test in config.get("tests", []):
            inputs = test.get("inputs", {})
            prompt = inputs.get("prompt", "")
            expectations = test.get("expectations", {})
            if not prompt:
                continue
            print(f"Calling {endpoint} with prompt: {prompt[:80]}...")
            generated = prompt_backend(prompt, endpoint)
            eval_data.append({
                "inputs":       inputs,
                "outputs":      generated,
                "expectations": expectations,
            })

        if not eval_data:
            print(f"No test cases in {config_path}, skipping.")
            continue

        # Evaluate with MLflow
        print(f"Running MLflow evaluate for {config_path} with {len(eval_data)} test(s)...")
        with mlflow.start_run(run_name=f"{config['name']}_{git_hash}"):
            mlflow.log_param("config_path", config_path)
            mlflow.log_param("endpoint", endpoint)
            mlflow.log_param("git_hash", git_hash)

            results = mlflow.genai.evaluate(
                data=eval_data,
                scorers=active_scorers,
            )

        print(f"Metrics for {config_path}: {results.metrics}")
        print(f"Results logged to MLflow. Tracking URI: {mlflow_tracking_uri}")


@dsl.pipeline(
    name="Canopy Eval (MLflow)",
    description="Pipeline for running canopy evals with MLflow scoring"
)
def canopy_eval_pipeline(
    repo_url: str,
    branch: str = "main",
    backend_url: str = "",
    llm_endpoint: str = "",
    mlflow_tracking_uri: str = "",
    git_hash: str = "test",
):
    eval_pvc = kubernetes.CreatePVC(
        pvc_name_suffix="-eval-pvc",
        access_modes=["ReadWriteOnce"],
        size="3Gi",
        storage_class_name="gp3-csi",
    )

    # Step 1: Clone repo
    clone_task = git_clone_op(repo_url=repo_url, branch=branch)
    kubernetes.mount_pvc(clone_task, pvc_name=eval_pvc.outputs["name"], mount_path="/prompts")
    kubernetes.use_secret_as_env(
        clone_task,
        secret_name="git-auth",
        secret_key_to_env={"username": "GIT_USERNAME", "password": "GIT_PASSWORD"},
    )

    # Step 2: Scan for test configs
    scan_task = scan_directory_op()
    scan_task.after(clone_task)
    kubernetes.mount_pvc(scan_task, pvc_name=eval_pvc.outputs["name"], mount_path="/prompts")

    # Step 3: Run evaluations
    test_task = run_all_mlflow_tests(
        configs=scan_task.outputs["configs"],
        backend_url=backend_url,
        llm_endpoint=llm_endpoint,
        mlflow_tracking_uri=mlflow_tracking_uri,
        git_hash=git_hash,
    )
    test_task.after(scan_task)
    kubernetes.mount_pvc(test_task, pvc_name=eval_pvc.outputs["name"], mount_path="/prompts")


if __name__ == "__main__":
    arguments = {
        "repo_url":             "https://<USER_NAME>:<PASSWORD>@<GIT_SERVER>/<USER_NAME>/evals.git",  # replace
        "branch":               "main",
        "backend_url":          "http://canopy-backend:8000",
        "llm_endpoint":         "http://llama-32-predictor.ai501.svc.cluster.local:80",
        "mlflow_tracking_uri":  "https://mlflow.redhat-ods-applications.svc.cluster.local:8443",
        "git_hash":             "test",
    }

    namespace_file_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    with open(namespace_file_path) as f:
        namespace = f.read()

    kubeflow_endpoint = f"https://ds-pipeline-dspa.{namespace}.svc:8443"

    sa_token_file_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    with open(sa_token_file_path) as f:
        bearer_token = f.read()

    ssl_ca_cert = "/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"

    print(f"Connecting to Data Science Pipelines: {kubeflow_endpoint}")
    client = kfp.Client(
        host=kubeflow_endpoint,
        existing_token=bearer_token,
        ssl_ca_cert=ssl_ca_cert,
    )

    client.create_run_from_pipeline_func(
        canopy_eval_pipeline,
        arguments=arguments,
        experiment_name="kfp-evals-pipeline",
        enable_caching=False,
    )

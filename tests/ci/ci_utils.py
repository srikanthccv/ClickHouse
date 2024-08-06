import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Union, Optional, Sequence

import requests


class Envs:
    GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY", "ClickHouse/ClickHouse")
    WORKFLOW_RESULT_FILE = os.getenv(
        "WORKFLOW_RESULT_FILE", "/tmp/workflow_results.json"
    )
    S3_BUILDS_BUCKET = os.getenv("S3_BUILDS_BUCKET", "clickhouse-builds")
    GITHUB_WORKFLOW = os.getenv("GITHUB_WORKFLOW", "")


class WithIter(type):
    def __iter__(cls):
        return (v for k, v in cls.__dict__.items() if not k.startswith("_"))


@contextmanager
def cd(path: Union[Path, str]) -> Iterator[None]:
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


class GH:
    class ActionsNames:
        RunConfig = "RunConfig"

    class ActionStatuses:
        ERROR = "error"
        FAILURE = "failure"
        PENDING = "pending"
        SUCCESS = "success"

    @classmethod
    def _get_workflow_results(cls):
        if not Path(Envs.WORKFLOW_RESULT_FILE).exists():
            print(
                f"ERROR: Failed to get workflow results from file [{Envs.WORKFLOW_RESULT_FILE}]"
            )
            return {}
        with open(Envs.WORKFLOW_RESULT_FILE, "r", encoding="utf-8") as json_file:
            try:
                res = json.load(json_file)
            except json.JSONDecodeError as e:
                print(f"ERROR: json decoder exception {e}")
                json_file.seek(0)
                print("    File content:")
                print(json_file.read())
                return {}
        return res

    @classmethod
    def print_workflow_results(cls):
        res = cls._get_workflow_results()
        results = [f"{job}: {data['result']}" for job, data in res.items()]
        cls.print_in_group("Workflow results", results)

    @classmethod
    def is_workflow_ok(cls) -> bool:
        res = cls._get_workflow_results()
        for _job, data in res.items():
            if data["result"] == "failure":
                return False
        return bool(res)

    @classmethod
    def get_workflow_job_result(cls, wf_job_name: str) -> Optional[str]:
        res = cls._get_workflow_results()
        if wf_job_name in res:
            return res[wf_job_name]["result"]  # type: ignore
        else:
            return None

    @staticmethod
    def print_in_group(group_name: str, lines: Union[Any, List[Any]]) -> None:
        lines = list(lines)
        print(f"::group::{group_name}")
        for line in lines:
            print(line)
        print("::endgroup::")

    @staticmethod
    def get_commit_status_by_name(
        token: str, commit_sha: str, status_name: Union[str, Sequence]
    ) -> str:
        assert len(token) == 40
        assert len(commit_sha) == 40
        assert Utils.is_hex(commit_sha)
        assert not Utils.is_hex(token)
        url = f"https://api.github.com/repos/{Envs.GITHUB_REPOSITORY}/commits/{commit_sha}/statuses?per_page={200}"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        response = requests.get(url, headers=headers, timeout=5)

        if isinstance(status_name, str):
            status_name = (status_name,)
        if response.status_code == 200:
            assert "next" not in response.links, "Response truncated"
            statuses = response.json()
            for status in statuses:
                if status["context"] in status_name:
                    return status["state"]  # type: ignore
        return ""

    @staticmethod
    def check_wf_completed(token: str, commit_sha: str) -> bool:
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        url = f"https://api.github.com/repos/{Envs.GITHUB_REPOSITORY}/commits/{commit_sha}/check-runs?per_page={100}"

        for i in range(3):
            try:
                response = requests.get(url, headers=headers, timeout=5)
                response.raise_for_status()
                # assert "next" not in response.links, "Response truncated"

                data = response.json()
                assert data["check_runs"], "?"

                for check in data["check_runs"]:
                    if check["status"] != "completed":
                        print(
                            f"   Check workflow status: Check not completed [{check['name']}]"
                        )
                        return False
                return True
            except Exception as e:
                print(f"ERROR: exception after attempt [{i}]: {e}")
                time.sleep(1)

        return False

    @staticmethod
    def get_pr_url_by_branch(branch, repo=None):
        repo = repo or Envs.GITHUB_REPOSITORY
        get_url_cmd = f"gh pr list --repo {repo} --head {branch} --json url --jq '.[0].url' --state open"
        url = Shell.get_output(get_url_cmd)
        if not url:
            print(f"WARNING: No open PR found, branch [{branch}] - search for merged")
            get_url_cmd = f"gh pr list --repo {repo} --head {branch} --json url --jq '.[0].url' --state merged"
            url = Shell.get_output(get_url_cmd)
        if not url:
            print(f"ERROR: PR nor found, branch [{branch}]")
        return url

    @staticmethod
    def is_latest_release_branch(branch):
        latest_branch = Shell.get_output(
            'gh pr list --label release --repo ClickHouse/ClickHouse --search "sort:created" -L1 --json headRefName'
        )
        return latest_branch == branch


class Shell:
    @classmethod
    def get_output_or_raise(cls, command):
        return cls.get_output(command, strict=True)

    @classmethod
    def get_output(cls, command, strict=False):
        res = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=strict,
        )
        return res.stdout.strip()

    @classmethod
    def check(
        cls,
        command,
        strict=False,
        verbose=False,
        dry_run=False,
        stdin_str=None,
        **kwargs,
    ):
        if dry_run:
            print(f"Dry-ryn. Would run command [{command}]")
            return True
        if verbose:
            print(f"Run command [{command}]")
        proc = subprocess.Popen(
            command,
            shell=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE if stdin_str else None,
            universal_newlines=True,
            start_new_session=True,
            bufsize=1,
            errors="backslashreplace",
            **kwargs,
        )
        if stdin_str:
            proc.communicate(input=stdin_str)
        elif proc.stdout:
            for line in proc.stdout:
                sys.stdout.write(line)
        proc.wait()
        if strict:
            assert proc.returncode == 0
        return proc.returncode == 0


class Utils:
    @staticmethod
    def get_failed_tests_number(description: str) -> Optional[int]:
        description = description.lower()

        pattern = r"fail:\s*(\d+)\s*(?=,|$)"
        match = re.search(pattern, description)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def is_killed_with_oom():
        if Shell.check(
            "sudo dmesg -T | grep -q -e 'Out of memory: Killed process' -e 'oom_reaper: reaped process' -e 'oom-kill:constraint=CONSTRAINT_NONE'"
        ):
            return True
        return False

    @staticmethod
    def clear_dmesg():
        Shell.check("sudo dmesg --clear", verbose=True)

    @staticmethod
    def is_hex(s):
        try:
            int(s, 16)
            return True
        except ValueError:
            return False

    @staticmethod
    def normalize_string(string: str) -> str:
        res = string.lower()
        for r in (
            (" ", "_"),
            ("(", "_"),
            (")", "_"),
            (",", "_"),
            ("/", "_"),
            ("-", "_"),
        ):
            res = res.replace(*r)
        return res

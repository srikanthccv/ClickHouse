#!/usr/bin/env python3

import csv
import os
import time
from typing import List
import logging

from ci_config import CI_CONFIG, REQUIRED_CHECKS
from env_helper import GITHUB_REPOSITORY, GITHUB_RUN_URL
from github import Github
from github.Commit import Commit
from pr_info import PRInfo, SKIP_MERGEABLE_CHECK_LABEL

RETRY = 5


def override_status(status: str, check_name: str, invert=False) -> str:
    if CI_CONFIG["tests_config"].get(check_name, {}).get("force_tests", False):
        return "success"

    if invert:
        if status == "success":
            return "error"
        return "success"

    return status


def get_commit(gh: Github, commit_sha: str, retry_count: int = RETRY) -> Commit:
    for i in range(retry_count):
        try:
            repo = gh.get_repo(GITHUB_REPOSITORY)
            commit = repo.get_commit(commit_sha)
            break
        except Exception as ex:
            if i == retry_count - 1:
                raise ex
            time.sleep(i)

    return commit


def post_commit_status(
    gh: Github, sha: str, check_name: str, description: str, state: str, report_url: str
):
    for i in range(RETRY):
        try:
            commit = get_commit(gh, sha, 1)
            commit.create_status(
                context=check_name,
                description=description,
                state=state,
                target_url=report_url,
            )
            break
        except Exception as ex:
            if i == RETRY - 1:
                raise ex
            time.sleep(i)


def post_commit_status_to_file(
    file_path: str, description: str, state: str, report_url: str
):
    if os.path.exists(file_path):
        raise Exception(f'File "{file_path}" already exists!')
    with open(file_path, "w", encoding="utf-8") as f:
        out = csv.writer(f, delimiter="\t")
        out.writerow([state, report_url, description])


def remove_labels(gh: Github, pr_info: PRInfo, labels_names: List[str]):
    repo = gh.get_repo(GITHUB_REPOSITORY)
    pull_request = repo.get_pull(pr_info.number)
    for label in labels_names:
        pull_request.remove_from_labels(label)


def post_labels(gh: Github, pr_info: PRInfo, labels_names: List[str]):
    repo = gh.get_repo(GITHUB_REPOSITORY)
    pull_request = repo.get_pull(pr_info.number)
    for label in labels_names:
        pull_request.add_to_labels(label)


def fail_mergeable_check(commit: Commit, description: str):
    commit.create_status(
        context="Mergeable Check",
        description=description,
        state="failure",
        target_url=GITHUB_RUN_URL,
    )


def reset_mergeable_check(commit: Commit, description: str = ""):
    commit.create_status(
        context="Mergeable Check",
        description=description,
        state="success",
        target_url=GITHUB_RUN_URL,
    )


def update_mergeable_check(gh: Github, pr_info: PRInfo, check_name: str):
    if SKIP_MERGEABLE_CHECK_LABEL in pr_info.labels:
        return

    logging.info("Update Mergeable Check by %s", check_name)

    commit = get_commit(gh, pr_info.sha)
    checks = {
        check.context: check.state
        for check in filter(
            lambda check: (check.context in REQUIRED_CHECKS),
            # get_statuses() returns generator, which cannot be reversed - we need comprehension
            # pylint: disable=unnecessary-comprehension
            reversed([status for status in commit.get_statuses()]),
        )
    }

    success = []
    fail = []
    for name, state in checks.items():
        if state == "success":
            success.append(name)
        else:
            fail.append(name)

    if fail:
        description = "failed: " + ", ".join(fail)
        if success:
            description += "; succeeded: " + ", ".join(success)
        if len(description) > 140:
            description = description[:137] + "..."
        fail_mergeable_check(commit, description)
        return

    description = ", ".join(success)
    if len(description) > 140:
        description = description[:137] + "..."
    reset_mergeable_check(commit, description)

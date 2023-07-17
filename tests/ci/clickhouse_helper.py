#!/usr/bin/env python3
from typing import List
import json
import logging
import time

import requests  # type: ignore

from get_robot_token import get_parameter_from_ssm
from pr_info import PRInfo
from report import TestResults


class InsertException(Exception):
    pass


class ClickHouseHelper:
    def __init__(self, url=None):
        if url is None:
            url = get_parameter_from_ssm("clickhouse-test-stat-url")

        self.url = url
        self.auth = {
            "X-ClickHouse-User": get_parameter_from_ssm("clickhouse-test-stat-login"),
            "X-ClickHouse-Key": get_parameter_from_ssm("clickhouse-test-stat-password"),
        }

    @staticmethod
    def _insert_json_str_info_impl(url, auth, db, table, json_str):
        params = {
            "database": db,
            "query": f"INSERT INTO {table} FORMAT JSONEachRow",
            "date_time_input_format": "best_effort",
            "send_logs_level": "warning",
        }

        for i in range(5):
            try:
                response = requests.post(
                    url, params=params, data=json_str, headers=auth
                )
            except Exception as e:
                error = f"Received exception while sending data to {url} on {i} attempt: {e}"
                logging.warning(error)
                continue

            logging.info("Response content '%s'", response.content)

            if response.ok:
                break

            error = (
                "Cannot insert data into clickhouse at try "
                + str(i)
                + ": HTTP code "
                + str(response.status_code)
                + ": '"
                + str(response.text)
                + "'"
            )

            if response.status_code >= 500:
                # A retriable error
                time.sleep(1)
                continue

            logging.info(
                "Request headers '%s', body '%s'",
                response.request.headers,
                response.request.body,
            )

            raise InsertException(error)
        else:
            raise InsertException(error)

    def _insert_json_str_info(self, db, table, json_str):
        self._insert_json_str_info_impl(self.url, self.auth, db, table, json_str)

    def insert_event_into(self, db, table, event, safe=True):
        event_str = json.dumps(event)
        try:
            self._insert_json_str_info(db, table, event_str)
        except InsertException as e:
            logging.error(
                "Exception happened during inserting data into clickhouse: %s", e
            )
            if not safe:
                raise

    def insert_events_into(self, db, table, events, safe=True):
        jsons = []
        for event in events:
            jsons.append(json.dumps(event))

        try:
            self._insert_json_str_info(db, table, ",".join(jsons))
        except InsertException as e:
            logging.error(
                "Exception happened during inserting data into clickhouse: %s", e
            )
            if not safe:
                raise

    def _select_and_get_json_each_row(self, db, query):
        params = {
            "database": db,
            "query": query,
            "default_format": "JSONEachRow",
        }
        for i in range(5):
            response = None
            try:
                response = requests.get(self.url, params=params, headers=self.auth)
                response.raise_for_status()
                return response.text
            except Exception as ex:
                logging.warning("Cannot insert with exception %s", str(ex))
                if response:
                    logging.warning("Reponse text %s", response.text)
                time.sleep(0.1 * i)

        raise Exception("Cannot fetch data from clickhouse")

    def select_json_each_row(self, db, query):
        text = self._select_and_get_json_each_row(db, query)
        result = []
        for line in text.split("\n"):
            if line:
                result.append(json.loads(line))
        return result


def prepare_tests_results_for_clickhouse(
    pr_info: PRInfo,
    test_results: TestResults,
    check_status: str,
    check_duration: float,
    check_start_time: str,
    report_url: str,
    check_name: str,
) -> List[dict]:
    pull_request_url = "https://github.com/ClickHouse/ClickHouse/commits/master"
    base_ref = "master"
    head_ref = "master"
    base_repo = pr_info.repo_full_name
    head_repo = pr_info.repo_full_name
    if pr_info.number != 0:
        pull_request_url = pr_info.pr_html_url
        base_ref = pr_info.base_ref
        base_repo = pr_info.base_name
        head_ref = pr_info.head_ref
        head_repo = pr_info.head_name

    common_properties = dict(
        pull_request_number=pr_info.number,
        commit_sha=pr_info.sha,
        commit_url=pr_info.commit_html_url,
        check_name=check_name,
        check_status=check_status,
        check_duration_ms=int(float(check_duration) * 1000),
        check_start_time=check_start_time,
        report_url=report_url,
        pull_request_url=pull_request_url,
        base_ref=base_ref,
        base_repo=base_repo,
        head_ref=head_ref,
        head_repo=head_repo,
        task_url=pr_info.task_url,
    )

    # Always publish a total record for all checks. For checks with individual
    # tests, also publish a record per test.
    result = [common_properties]
    for test_result in test_results:
        current_row = common_properties.copy()
        test_name = test_result.name
        test_status = test_result.status

        test_time = test_result.time or 0
        current_row["test_duration_ms"] = int(test_time * 1000)
        current_row["test_name"] = test_name
        current_row["test_status"] = test_status
        if test_result.raw_logs:
            # Protect from too big blobs that contain garbage
            current_row["test_context_raw"] = test_result.raw_logs[: 32 * 1024]
        else:
            current_row["test_context_raw"] = ""
        result.append(current_row)

    return result

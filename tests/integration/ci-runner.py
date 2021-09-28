#!/usr/bin/env python3

import logging
import subprocess
import os
import glob
import time
import shutil
from collections import defaultdict
import random
import json
import csv


MAX_RETRY = 3
NUM_WORKERS = 5
SLEEP_BETWEEN_RETRIES = 5
PARALLEL_GROUP_SIZE = 100
CLICKHOUSE_BINARY_PATH = "/usr/bin/clickhouse"
CLICKHOUSE_ODBC_BRIDGE_BINARY_PATH = "/usr/bin/clickhouse-odbc-bridge"
CLICKHOUSE_LIBRARY_BRIDGE_BINARY_PATH = "/usr/bin/clickhouse-library-bridge"

TRIES_COUNT = 10
MAX_TIME_SECONDS = 3600

MAX_TIME_IN_SANDBOX = 20 * 60   # 20 minutes
TASK_TIMEOUT = 8 * 60 * 60      # 8 hours

def get_tests_to_run(pr_info):
    result = set([])
    changed_files = pr_info['changed_files']

    if changed_files is None:
        return []

    for fpath in changed_files:
        if 'tests/integration/test_' in fpath:
            logging.info('File %s changed and seems like integration test', fpath)
            result.add(fpath.split('/')[2])
    return list(result)


def filter_existing_tests(tests_to_run, repo_path):
    result = []
    for relative_test_path in tests_to_run:
        if os.path.exists(os.path.join(repo_path, 'tests/integration', relative_test_path)):
            result.append(relative_test_path)
        else:
            logging.info("Skipping test %s, seems like it was removed", relative_test_path)
    return result


def _get_deselect_option(tests):
    return ' '.join(['--deselect {}'.format(t) for t in tests])

# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_test_results_output(fname):
    read = False
    description_output = []
    with open(fname, 'r') as out:
        for line in out:
            if read and line.strip() and not line.startswith('=='):
                description_output.append(line.strip())
            if 'short test summary info' in line:
                read = True
    return description_output


def get_counters(output):
    counters = {
        "ERROR": set([]),
        "PASSED": set([]),
        "FAILED": set([]),
    }

    for line in output:
        if '.py' in line:
            line_arr = line.strip().split(' ')
            state = line_arr[0]
            test_name = ' '.join(line_arr[1:])
            if ' - ' in test_name:
                test_name = test_name[:test_name.find(' - ')]
            if state in counters:
                counters[state].add(test_name)
            else:
                logging.info("Strange line %s", line)
        else:
            logging.info("Strange line %s", line)
    return {k: list(v) for k, v in counters.items()}


def parse_test_times(fname):
    read = False
    description_output = []
    with open(fname, 'r') as out:
        for line in out:
            if read and '==' in line:
                break
            if read and line.strip():
                description_output.append(line.strip())
            if 'slowest durations' in line:
                read = True
    return description_output


def get_test_times(output):
    result = defaultdict(float)
    for line in output:
        if '.py' in line:
            line_arr = line.strip().split(' ')
            test_time = line_arr[0]
            test_name = ' '.join([elem for elem in line_arr[2:] if elem])
            if test_name not in result:
                result[test_name] = 0.0
            result[test_name] += float(test_time[:-1])
    return result


def clear_ip_tables_and_restart_daemons():
    logging.info("Dump iptables after run %s", subprocess.check_output("iptables -L", shell=True))
    try:
        logging.info("Killing all alive docker containers")
        subprocess.check_output("timeout -s 9 10m docker kill $(docker ps -q)", shell=True)
    except subprocess.CalledProcessError as err:
        logging.info("docker kill excepted: " + str(err))

    try:
        logging.info("Removing all docker containers")
        subprocess.check_output("timeout -s 9 10m docker rm $(docker ps -a -q) --force", shell=True)
    except subprocess.CalledProcessError as err:
        logging.info("docker rm excepted: " + str(err))

    try:
        logging.info("Stopping docker daemon")
        subprocess.check_output("service docker stop", shell=True)
    except subprocess.CalledProcessError as err:
        logging.info("docker stop excepted: " + str(err))

    try:
        for i in range(200):
            try:
                logging.info("Restarting docker %s", i)
                subprocess.check_output("service docker start", shell=True)
                subprocess.check_output("docker ps", shell=True)
                break
            except subprocess.CalledProcessError as err:
                time.sleep(0.5)
                logging.info("Waiting docker to start, current %s", str(err))
        else:
            raise Exception("Docker daemon doesn't responding")
    except subprocess.CalledProcessError as err:
        logging.info("Can't reload docker: " + str(err))

    iptables_iter = 0
    try:
        for i in range(1000):
            iptables_iter = i
            # when rules will be empty, it will raise exception
            subprocess.check_output("iptables -D DOCKER-USER 1", shell=True)
    except subprocess.CalledProcessError as err:
        logging.info("All iptables rules cleared, " + str(iptables_iter) + "iterations, last error: " + str(err))


class ClickhouseIntegrationTestsRunner:

    def __init__(self, result_path, params):
        self.result_path = result_path
        self.params = params

        self.image_versions = self.params['docker_images_with_versions']
        self.shuffle_groups = self.params['shuffle_test_groups']
        self.flaky_check = 'flaky check' in self.params['context_name']
        self.start_time = time.time()
        self.soft_deadline_time = self.start_time + (TASK_TIMEOUT - MAX_TIME_IN_SANDBOX)

    def path(self):
        return self.result_path

    def base_path(self):
        return os.path.join(str(self.result_path), '../')

    def should_skip_tests(self):
        return []

    def get_image_with_version(self, name):
        if name in self.image_versions:
            return name + ":" + self.image_versions[name]
        logging.warn("Cannot find image %s in params list %s", name, self.image_versions)
        if ':' not in name:
            return name + ":latest"
        return name

    def get_single_image_version(self):
        name = self.get_images_names()[0]
        if name in self.image_versions:
            return self.image_versions[name]
        logging.warn("Cannot find image %s in params list %s", name, self.image_versions)
        return 'latest'

    def shuffle_test_groups(self):
        return self.shuffle_groups != 0

    @staticmethod
    def get_images_names():
        return ["clickhouse/integration-tests-runner", "clickhouse/mysql-golang-client",
                "clickhouse/mysql-java-client", "clickhouse/mysql-js-client",
                "clickhouse/mysql-php-client", "clickhouse/postgresql-java-client",
                "clickhouse/integration-test", "clickhouse/kerberos-kdc",
                "clickhouse/integration-helper", ]


    def _can_run_with(self, path, opt):
        with open(path, 'r') as script:
            for line in script:
                if opt in line:
                    return True
        return False

    def _install_clickhouse(self, debs_path):
        for package in ('clickhouse-common-static_', 'clickhouse-server_', 'clickhouse-client', 'clickhouse-common-static-dbg_'):  # order matters
            logging.info("Installing package %s", package)
            for f in os.listdir(debs_path):
                if package in f:
                    full_path = os.path.join(debs_path, f)
                    logging.info("Package found in %s", full_path)
                    log_name = "install_" + f + ".log"
                    log_path = os.path.join(str(self.path()), log_name)
                    with open(log_path, 'w') as log:
                        cmd = "dpkg -i {}".format(full_path)
                        logging.info("Executing installation cmd %s", cmd)
                        retcode = subprocess.Popen(cmd, shell=True, stderr=log, stdout=log).wait()
                        if retcode == 0:
                            logging.info("Instsallation of %s successfull", full_path)
                        else:
                            raise Exception("Installation of %s failed", full_path)
                    break
            else:
                raise Exception("Package with {} not found".format(package))
        logging.info("Unstripping binary")
        # logging.info("Unstring %s", subprocess.check_output("eu-unstrip /usr/bin/clickhouse {}".format(CLICKHOUSE_BINARY_PATH), shell=True))

        logging.info("All packages installed")
        os.chmod(CLICKHOUSE_BINARY_PATH, 0o777)
        os.chmod(CLICKHOUSE_ODBC_BRIDGE_BINARY_PATH, 0o777)
        os.chmod(CLICKHOUSE_LIBRARY_BRIDGE_BINARY_PATH, 0o777)
        result_path_bin = os.path.join(str(self.base_path()), "clickhouse")
        result_path_odbc_bridge = os.path.join(str(self.base_path()), "clickhouse-odbc-bridge")
        result_path_library_bridge = os.path.join(str(self.base_path()), "clickhouse-library-bridge")
        shutil.copy(CLICKHOUSE_BINARY_PATH, result_path_bin)
        shutil.copy(CLICKHOUSE_ODBC_BRIDGE_BINARY_PATH, result_path_odbc_bridge)
        shutil.copy(CLICKHOUSE_LIBRARY_BRIDGE_BINARY_PATH, result_path_library_bridge)
        return None, None

    def _compress_logs(self, dir, relpaths, result_path):
        subprocess.check_call("tar czf {} -C {} {}".format(result_path, dir, ' '.join(relpaths)), shell=True)  # STYLE_CHECK_ALLOW_SUBPROCESS_CHECK_CALL

    def _get_all_tests(self, repo_path):
        image_cmd = self._get_runner_image_cmd(repo_path)
        out_file = "all_tests.txt"
        out_file_full = "all_tests_full.txt"
        cmd = "cd {repo_path}/tests/integration && " \
            "timeout -s 9 1h ./runner --tmpfs {image_cmd} ' --setup-plan' " \
            "| tee {out_file_full} | grep '::' | sed 's/ (fixtures used:.*//g' | sed 's/^ *//g' | sed 's/ *$//g' " \
            "| grep -v 'SKIPPED' | sort -u  > {out_file}".format(
                repo_path=repo_path, image_cmd=image_cmd, out_file=out_file, out_file_full=out_file_full)

        logging.info("Getting all tests with cmd '%s'", cmd)
        subprocess.check_call(cmd, shell=True)  # STYLE_CHECK_ALLOW_SUBPROCESS_CHECK_CALL

        all_tests_file_path = "{repo_path}/tests/integration/{out_file}".format(repo_path=repo_path, out_file=out_file)
        if not os.path.isfile(all_tests_file_path) or os.path.getsize(all_tests_file_path) == 0:
            all_tests_full_file_path = "{repo_path}/tests/integration/{out_file}".format(repo_path=repo_path, out_file=out_file_full)
            if os.path.isfile(all_tests_full_file_path):
                # log runner output
                logging.info("runner output:")
                with open(all_tests_full_file_path, 'r') as all_tests_full_file:
                    for line in all_tests_full_file:
                        line = line.rstrip()
                        if line:
                            logging.info("runner output: %s", line)
            else:
                logging.info("runner output '%s' is empty", all_tests_full_file_path)

            raise Exception("There is something wrong with getting all tests list: file '{}' is empty or does not exist.".format(all_tests_file_path))

        all_tests = []
        with open(all_tests_file_path, "r") as all_tests_file:
            for line in all_tests_file:
                all_tests.append(line.strip())
        return list(sorted(all_tests))

    def _get_parallel_tests_skip_list(self, repo_path):
        skip_list_file_path = "{}/tests/integration/parallel_skip.json".format(repo_path)
        if not os.path.isfile(skip_list_file_path) or os.path.getsize(skip_list_file_path) == 0:
            raise Exception("There is something wrong with getting all tests list: file '{}' is empty or does not exist.".format(skip_list_file_path))

        skip_list_tests = []
        with open(skip_list_file_path, "r") as skip_list_file:
            skip_list_tests = json.load(skip_list_file)
        return list(sorted(skip_list_tests))

    def group_test_by_file(self, tests):
        result = {}
        for test in tests:
            test_file = test.split('::')[0]
            if test_file not in result:
                result[test_file] = []
            result[test_file].append(test)
        return result

    def _update_counters(self, main_counters, current_counters):
        for test in current_counters["PASSED"]:
            if test not in main_counters["PASSED"] and test not in main_counters["FLAKY"]:
                is_flaky = False
                if test in main_counters["FAILED"]:
                    main_counters["FAILED"].remove(test)
                    is_flaky = True
                if test in main_counters["ERROR"]:
                    main_counters["ERROR"].remove(test)
                    is_flaky = True
                if is_flaky:
                    main_counters["FLAKY"].append(test)
                else:
                    main_counters["PASSED"].append(test)

        for state in ("ERROR", "FAILED"):
            for test in current_counters[state]:
                if test in main_counters["FLAKY"]:
                    continue
                if test in main_counters["PASSED"]:
                    main_counters["PASSED"].remove(test)
                    main_counters["FLAKY"].append(test)
                    continue
                if test not in main_counters[state]:
                    main_counters[state].append(test)

    def _get_runner_image_cmd(self, repo_path):
        image_cmd = ''
        if self._can_run_with(os.path.join(repo_path, "tests/integration", "runner"), '--docker-image-version'):
            for img in self.get_images_names():
                if img == "clickhouse/integration-tests-runner":
                    runner_version = self.get_single_image_version()
                    logging.info("Can run with custom docker image version %s", runner_version)
                    image_cmd += ' --docker-image-version={} '.format(runner_version)
                else:
                    if self._can_run_with(os.path.join(repo_path, "tests/integration", "runner"), '--docker-compose-images-tags'):
                        image_cmd += '--docker-compose-images-tags={} '.format(self.get_image_with_version(img))
        else:
            image_cmd = ''
            logging.info("Cannot run with custom docker image version :(")
        return image_cmd

    def _find_test_data_dirs(self, repo_path, test_names):
        relpaths = {}
        for test_name in test_names:
            if '/' in test_name:
                test_dir = test_name[:test_name.find('/')]
            else:
                test_dir = test_name
            if os.path.isdir(os.path.join(repo_path, "tests/integration", test_dir)):
                for name in os.listdir(os.path.join(repo_path, "tests/integration", test_dir)):
                    relpath = os.path.join(os.path.join(test_dir, name))
                    mtime = os.path.getmtime(os.path.join(repo_path, "tests/integration", relpath))
                    relpaths[relpath] = mtime
        return relpaths

    def _get_test_data_dirs_difference(self, new_snapshot, old_snapshot):
        res = set()
        for path in new_snapshot:
            if (not path in old_snapshot) or (old_snapshot[path] != new_snapshot[path]):
                res.add(path)
        return res

    def try_run_test_group(self, repo_path, test_group, tests_in_group, num_tries, num_workers):
        try:
            return self.run_test_group(repo_path, test_group, tests_in_group, num_tries, num_workers)
        except Exception as e:
            logging.info("Failed to run {}:\n{}".format(str(test_group), str(e)))
            counters = {
                "ERROR": [],
                "PASSED": [],
                "FAILED": [],
                "SKIPPED": [],
                "FLAKY": [],
            }
            tests_times = defaultdict(float)
            for test in tests_in_group:
                counters["ERROR"].append(test)
                tests_times[test] = 0
            return counters, tests_times, []

    def run_test_group(self, repo_path, test_group, tests_in_group, num_tries, num_workers):
        counters = {
            "ERROR": [],
            "PASSED": [],
            "FAILED": [],
            "SKIPPED": [],
            "FLAKY": [],
        }
        tests_times = defaultdict(float)

        if self.soft_deadline_time < time.time():
            for test in tests_in_group:
                logging.info("Task timeout exceeded, skipping %s", test)
                counters["SKIPPED"].append(test)
                tests_times[test] = 0
            return counters, tests_times, []

        image_cmd = self._get_runner_image_cmd(repo_path)
        test_group_str = test_group.replace('/', '_').replace('.', '_')

        log_paths = []
        test_data_dirs = {}

        for i in range(num_tries):
            logging.info("Running test group %s for the %s retry", test_group, i)
            clear_ip_tables_and_restart_daemons()

            test_names = set([])
            for test_name in tests_in_group:
                if test_name not in counters["PASSED"]:
                    if '[' in test_name:
                        test_names.add(test_name[:test_name.find('[')])
                    else:
                        test_names.add(test_name)

            if i == 0:
                test_data_dirs = self._find_test_data_dirs(repo_path, test_names)

            info_basename = test_group_str + "_" + str(i) + ".nfo"
            info_path = os.path.join(repo_path, "tests/integration", info_basename)

            test_cmd = ' '.join([test for test in sorted(test_names)])
            parallel_cmd = " --parallel {} ".format(num_workers) if num_workers > 0 else ""
            cmd = "cd {}/tests/integration && timeout -s 9 1h ./runner --tmpfs {} -t {} {} '-rfEp --run-id={} --color=no --durations=0 {}' | tee {}".format(
                repo_path, image_cmd, test_cmd, parallel_cmd, i, _get_deselect_option(self.should_skip_tests()), info_path)

            log_basename = test_group_str + "_" + str(i) + ".log"
            log_path = os.path.join(repo_path, "tests/integration", log_basename)
            with open(log_path, 'w') as log:
                logging.info("Executing cmd: %s", cmd)
                retcode = subprocess.Popen(cmd, shell=True, stderr=log, stdout=log).wait()
                if retcode == 0:
                    logging.info("Run %s group successfully", test_group)
                else:
                    logging.info("Some tests failed")

            extra_logs_names = [log_basename]
            log_result_path = os.path.join(str(self.path()), 'integration_run_' + log_basename)
            shutil.copy(log_path, log_result_path)
            log_paths.append(log_result_path)

            for pytest_log_path in glob.glob(os.path.join(repo_path, "tests/integration/pytest*.log")):
                new_name = test_group_str + "_" + str(i) + "_" + os.path.basename(pytest_log_path)
                os.rename(pytest_log_path, os.path.join(repo_path, "tests/integration", new_name))
                extra_logs_names.append(new_name)

            dockerd_log_path = os.path.join(repo_path, "tests/integration/dockerd.log")
            if os.path.exists(dockerd_log_path):
                new_name = test_group_str + "_" + str(i) + "_" + os.path.basename(dockerd_log_path)
                os.rename(dockerd_log_path, os.path.join(repo_path, "tests/integration", new_name))
                extra_logs_names.append(new_name)

            if os.path.exists(info_path):
                extra_logs_names.append(info_basename)
                lines = parse_test_results_output(info_path)
                new_counters = get_counters(lines)
                times_lines = parse_test_times(info_path)
                new_tests_times = get_test_times(times_lines)
                self._update_counters(counters, new_counters)
                for test_name, test_time in new_tests_times.items():
                    tests_times[test_name] = test_time

            test_data_dirs_new = self._find_test_data_dirs(repo_path, test_names)
            test_data_dirs_diff = self._get_test_data_dirs_difference(test_data_dirs_new, test_data_dirs)
            test_data_dirs = test_data_dirs_new

            if extra_logs_names or test_data_dirs_diff:
                extras_result_path = os.path.join(str(self.path()), "integration_run_" + test_group_str + "_" + str(i) + ".tar.gz")
                self._compress_logs(os.path.join(repo_path, "tests/integration"), extra_logs_names + list(test_data_dirs_diff), extras_result_path)
                log_paths.append(extras_result_path)

            if len(counters["PASSED"]) + len(counters["FLAKY"]) == len(tests_in_group):
                logging.info("All tests from group %s passed", test_group)
                break
            if len(counters["PASSED"]) + len(counters["FLAKY"]) >= 0 and len(counters["FAILED"]) == 0 and len(counters["ERROR"]) == 0:
                logging.info("Seems like all tests passed but some of them are skipped or deselected. Ignoring them and finishing group.")
                break
        else:
            # Mark all non tried tests as errors, with '::' in name
            # (example test_partition/test.py::test_partition_simple). For flaky check
            # we run whole test dirs like "test_odbc_interaction" and don't
            # want to mark them as error so we filter by '::'.
            for test in tests_in_group:
                if (test not in counters["PASSED"] and
                    test not in counters["ERROR"] and
                    test not in counters["FAILED"] and
                    '::' in test):
                    counters["ERROR"].append(test)

        return counters, tests_times, log_paths

    def run_flaky_check(self, repo_path, build_path):
        pr_info = self.params['pr_info']

        # pytest swears, if we require to run some tests which was renamed or deleted
        tests_to_run = filter_existing_tests(get_tests_to_run(pr_info), repo_path)
        if not tests_to_run:
            logging.info("No tests to run found")
            return 'success', 'Nothing to run', [('Nothing to run', 'OK')], ''

        self._install_clickhouse(build_path)
        logging.info("Found '%s' tests to run", ' '.join(tests_to_run))
        result_state = "success"
        description_prefix = "No flaky tests: "
        start = time.time()
        logging.info("Starting check with retries")
        final_retry = 0
        logs = []
        for i in range(TRIES_COUNT):
            final_retry += 1
            logging.info("Running tests for the %s time", i)
            counters, tests_times, log_paths = self.try_run_test_group(repo_path, "flaky", tests_to_run, 1, 1)
            logs += log_paths
            if counters["FAILED"]:
                logging.info("Found failed tests: %s", ' '.join(counters["FAILED"]))
                description_prefix = "Flaky tests found: "
                result_state = "failure"
                break
            if counters["ERROR"]:
                description_prefix = "Flaky tests found: "
                logging.info("Found error tests: %s", ' '.join(counters["ERROR"]))
                # NOTE "error" result state will restart the whole test task, so we use "failure" here
                result_state = "failure"
                break
            assert len(counters["FLAKY"]) == 0
            logging.info("Try is OK, all tests passed, going to clear env")
            clear_ip_tables_and_restart_daemons()
            logging.info("And going to sleep for some time")
            if time.time() - start > MAX_TIME_SECONDS:
                logging.info("Timeout reached, going to finish flaky check")
                break
            time.sleep(5)

        test_result = []
        for state in ("ERROR", "FAILED", "PASSED", "SKIPPED", "FLAKY"):
            if state == "PASSED":
                text_state = "OK"
            elif state == "FAILED":
                text_state = "FAIL"
            else:
                text_state = state
            test_result += [(c + ' (✕' + str(final_retry) + ')', text_state, "{:.2f}".format(tests_times[c])) for c in counters[state]]
        status_text = description_prefix + ', '.join([str(n).lower().replace('failed', 'fail') + ': ' + str(len(c)) for n, c in counters.items()])

        return result_state, status_text, test_result, logs

    def run_impl(self, repo_path, build_path):
        if self.flaky_check:
            return self.run_flaky_check(repo_path, build_path)

        self._install_clickhouse(build_path)
        logging.info("Dump iptables before run %s", subprocess.check_output("iptables -L", shell=True))
        all_tests = self._get_all_tests(repo_path)
        parallel_skip_tests = self._get_parallel_tests_skip_list(repo_path)
        logging.info("Found %s tests first 3 %s", len(all_tests), ' '.join(all_tests[:3]))
        filtered_sequential_tests = list(filter(lambda test: test in all_tests, parallel_skip_tests))
        filtered_parallel_tests = list(filter(lambda test: test not in parallel_skip_tests, all_tests))
        not_found_tests = list(filter(lambda test: test not in all_tests, parallel_skip_tests))
        logging.info("Found %s tests first 3 %s, parallel %s, other %s", len(all_tests), ' '.join(all_tests[:3]), len(filtered_parallel_tests), len(filtered_sequential_tests))
        logging.info("Not found %s tests first 3 %s", len(not_found_tests), ' '.join(not_found_tests[:3]))

        grouped_tests = self.group_test_by_file(filtered_sequential_tests)
        i = 0
        for par_group in chunks(filtered_parallel_tests, PARALLEL_GROUP_SIZE):
            grouped_tests["parallel{}".format(i)] = par_group
            i+=1
        logging.info("Found %s tests groups", len(grouped_tests))

        counters = {
            "ERROR": [],
            "PASSED": [],
            "FAILED": [],
            "SKIPPED": [],
            "FLAKY": [],
        }
        tests_times = defaultdict(float)
        tests_log_paths = defaultdict(list)

        items_to_run = list(grouped_tests.items())

        logging.info("Total test groups %s", len(items_to_run))
        if self.shuffle_test_groups():
            logging.info("Shuffling test groups")
            random.shuffle(items_to_run)

        for group, tests in items_to_run:
            logging.info("Running test group %s countaining %s tests", group, len(tests))
            group_counters, group_test_times, log_paths = self.try_run_test_group(repo_path, group, tests, MAX_RETRY, NUM_WORKERS)
            total_tests = 0
            for counter, value in group_counters.items():
                logging.info("Tests from group %s stats, %s count %s", group, counter, len(value))
                counters[counter] += value
                logging.info("Totally have %s with status %s", len(counters[counter]), counter)
                total_tests += len(counters[counter])
            logging.info("Totally finished tests %s/%s", total_tests, len(all_tests))

            for test_name, test_time in group_test_times.items():
                tests_times[test_name] = test_time
                tests_log_paths[test_name] = log_paths

            if len(counters["FAILED"]) + len(counters["ERROR"]) >= 20:
                logging.info("Collected more than 20 failed/error tests, stopping")
                break

        if counters["FAILED"] or counters["ERROR"]:
            logging.info("Overall status failure, because we have tests in FAILED or ERROR state")
            result_state = "failure"
        else:
            logging.info("Overall success!")
            result_state = "success"

        test_result = []
        for state in ("ERROR", "FAILED", "PASSED", "SKIPPED", "FLAKY"):
            if state == "PASSED":
                text_state = "OK"
            elif state == "FAILED":
                text_state = "FAIL"
            else:
                text_state = state
            test_result += [(c, text_state, "{:.2f}".format(tests_times[c]), tests_log_paths[c]) for c in counters[state]]

        failed_sum = len(counters['FAILED']) + len(counters['ERROR'])
        status_text = "fail: {}, passed: {}, flaky: {}".format(failed_sum, len(counters['PASSED']), len(counters['FLAKY']))

        if self.soft_deadline_time < time.time():
            status_text = "Timeout, " + status_text
            result_state = "failure"

        counters['FLAKY'] = []
        if not counters or sum(len(counter) for counter in counters.values()) == 0:
            status_text = "No tests found for some reason! It's a bug"
            result_state = "failure"

        if '(memory)' in self.params['context_name']:
            result_state = "success"

        return result_state, status_text, test_result, []

def write_results(results_file, status_file, results, status):
    with open(results_file, 'w') as f:
        out = csv.writer(f, delimiter='\t')
        out.writerows(results)
    with open(status_file, 'w') as f:
        out = csv.writer(f, delimiter='\t')
        out.writerow(status)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    repo_path = os.environ.get("CLICKHOUSE_TESTS_REPO_PATH")
    build_path = os.environ.get("CLICKHOUSE_TESTS_BUILD_PATH")
    result_path = os.environ.get("CLICKHOUSE_TESTS_RESULT_PATH")
    params_path = os.environ.get("CLICKHOUSE_TESTS_JSON_PARAMS_PATH")

    params = json.loads(open(params_path, 'r').read())
    runner = ClickhouseIntegrationTestsRunner(result_path, params)

    logging.info("Running tests")
    state, description, test_results, _ = runner.run_impl(repo_path, build_path)
    logging.info("Tests finished")

    status = (state, description)
    out_results_file = os.path.join(str(runner.path()), "test_results.tsv")
    out_status_file = os.path.join(str(runner.path()), "check_status.tsv")
    write_results(out_results_file, out_status_file, test_results, status)
    logging.info("Result written")

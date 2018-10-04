#!/usr/bin/env python
#-*- coding: utf-8 -*-
import subprocess
import os
import argparse
import logging

CURRENT_FOLDER = os.path.dirname(os.path.realpath(__file__))

IMAGE_MAP = {
    "deb": "yandex/clickhouse-deb-builder",
    "binary": "yandex/clickhouse-binary-builder",
}


def check_image_exists_locally(image_name):
    try:
        output = subprocess.check_output("docker images -q {} 2> /dev/null".format(image_name), shell=True)
        return output != ""
    except subprocess.CalledProcessError as ex:
        logging.info("Image not found locally, error %s", str(ex))
        return False

def pull_image(image_name):
    try:
        subprocess.check_call("docker pull {}".format(image_name), shell=True)
        return True
    except subprocess.CalledProcessError as ex:
        logging.info("Cannot pull image %s, error %s", image_name, str(ex))
        return False

def build_image(image_name, filepath):
    subprocess.check_call("docker build --network=host -t {} -f {} .".format(image_name, filepath), shell=True)

def run_image_with_env(image_name, output, env_variables):
    env_part = " -e ".join(env_variables)
    if env_part:
        env_part = " -e " + env_part

    cmd = "docker run --network=host --rm --volume={output_path}:/output --volume={ch_root}:/build {env} -it {img_name}".format(
        output_path=output,
        ch_root=os.path.abspath(os.path.join(CURRENT_FOLDER, "../../")),
        env=env_part,
        img_name=image_name
    )

    logging.info("Will build ClickHouse pkg with cmd: '%s'", cmd)

    subprocess.check_call(cmd, shell=True)

def parse_env_variables(build_type, compiler, sanitizer, package_type):
    result = []
    if package_type == "deb":
        result.append("DEB_CC={}".format(compiler))
        result.append("DEB_CXX={}".format(compiler.replace('gcc', 'g++').replace('clang', 'clang++')))
    elif package_type == "binary":
        result.append("CC={}".format(compiler))
        result.append("CXX={}".format(compiler.replace('gcc', 'g++').replace('clang', 'clang++')))

    if sanitizer:
        result.append("SANITIZER={}".format(sanitizer))
    if build_type:
        result.append("BUILD_TYPE={}".format(build_type))


    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser(description="ClickHouse building script via docker")
    parser.add_argument("--package-type", choices=IMAGE_MAP.keys())
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--build-type", choices=("debug", ""), default="")
    parser.add_argument("--compiler", choices=("clang-6.0", "gcc-7", "gcc-8"), default="gcc-7")
    parser.add_argument("--sanitizer", choices=("address", "thread", "memory", "undefined", ""), default="")
    parser.add_argument("--force-build-image", action="store_true")

    args = parser.parse_args()
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(os.path.join(CURRENT_FOLDER, args.output_dir))

    image_name = IMAGE_MAP[args.package_type]
    dockerfile = os.path.join(CURRENT_FOLDER, args.package_type, "Dockerfile")
    try:
        if args.force_build_image or not check_image_exists_locally(image_name):
            if args.force_build_image or not pull_image(image_name):
                build_image(image_name, dockerfile)
        run_image_with_env(image_name, args.output_dir, parse_env_variables(args.build_type, args.compiler, args.sanitizer, args.package_type))
        logging.info("Output placed into %s", args.output_dir)
    except Exception as ex:
        logging.info("Exception occured %s", str(ex))

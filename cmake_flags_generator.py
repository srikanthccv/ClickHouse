import re
import os.path
from typing import TextIO, List, Tuple, Optional

Entity = Tuple[str, str, str]

# https://regex101.com/r/R6iogw/11
cmake_option_regex: str = r"^\s*option\s*\(([A-Z_0-9${}]+)\s*(?:\"((?:.|\n)*?)\")?\s*(.*)?\).*$"

output_file_name: str = "cmake_flags_and_output.md"
header_file_name: str = "cmake_files_header.md"
footer_file_name: str = "cmake_files_footer.md"

ch_master_url: str = "https://github.com/clickhouse/clickhouse/blob/master/"

name_str: str = "<a name=\"{anchor}\"></a>[`{name}`](" + ch_master_url + "{path}#L{line})"
default_anchor_str: str = "[`{name}`](#{anchor})"

def build_entity(path: str, entity: Entity, line_comment: Tuple[int, str], **options) -> str:
    (line, comment) = line_comment
    (_name, _description, default) = entity

    def make_anchor(t: str) -> str:
        return "".join(["-" if i == "_" else i.lower() for i in t if i.isalpha() or i == "_"])

    if len(default) == 0:
        default = "`OFF`"
    elif default[0] == "$":
        default = default[2:-1]
        default = default_anchor_str.format(
            name=default,
            anchor=make_anchor(default))
    else:
        default = "`" + default + "`"

    name: str = name_str.format(
        anchor=make_anchor(_name),
        name=_name,
        path=path,
        line=line)

    if options.get("no_desc", False):
        description: str = ""
    else:
        description: str = "".join(_description.split("\n")) + " | "

    return "| " + name + " | " + default + " | " + description + comment + " |"

def process_file(input_name: str, **options) -> List[str]:
    out: List[str] = []

    with open(input_name, 'r') as cmake_file:
        contents: str = cmake_file.read()

        def get_line_and_comment(target: str) -> Tuple[int, str]:
            contents_list: List[str] = contents.split("\n")
            comment: str = ""

            for n, line in enumerate(contents_list):
                if line.find(target) == -1:
                    continue

                for maybe_comment_line in contents_list[n - 1::-1]:
                    if not re.match("\s*#\s*", maybe_comment_line):
                        break

                    comment = re.sub("\s*#\s*", "", maybe_comment_line) + ". " + comment

                return n, comment

        matches: Optional[List[Entity]] = re.findall(cmake_option_regex, contents, re.MULTILINE)

        if matches:
            for entity in matches:
                out.append(
                    build_entity(
                        input_name,
                        entity,
                        get_line_and_comment(entity[0]),
                    **options))

    return out

def write_file(output: TextIO, in_file_name: str, **options) -> None:
    output.write("\n".join(sorted(process_file(in_file_name, **options))))

def process_folder(output: TextIO, name: str) -> None:
    for root, _, files in os.walk(name):
        for f in files:
            if f == "CMakeLists.txt" or ".cmake" in f:
                write_file(output, root + "/" + f)

def process() -> None:
    with open(output_file_name, "w") as f:
        with open(header_file_name, "r") as header:
            f.write(header.read())

        write_file(f, "CMakeLists.txt")
        write_file(f, "PreLoad.cmake")

        process_folder(f, "base")
        process_folder(f, "cmake")
        process_folder(f, "src")

        # Various ClickHouse extern parts (Copier/Obfuscator/...)

        f.write("""

### ClickHouse additory parts
| Name | Default value | Description |
|------|---------------|-------------|
""")

        write_file(f, "programs/CMakeLists.txt", no_desc=True)

        with open(footer_file_name, "r") as footer:
            f.write(footer.read())

process()

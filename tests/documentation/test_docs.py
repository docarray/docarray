import pathlib

import pytest
from mktestdocs import grab_code_blocks
from mktestdocs.__main__ import _executors, check_raw_string

from tests.index.elastic.fixture import start_storage_v8  # noqa: F401

file_to_skip = ['fastAPI', 'jina', 'index', 'first_steps.md']


def check_raw_file_full(raw, lang="python", keyword_ignore=[]):
    if lang not in _executors:
        raise LookupError(
            f"{lang} is not a supported language to check\n"
            "\tHint: you can add support for any language by using register_executor"
        )
    executor = _executors[lang]
    all_code = ""
    add_code_block = True

    for b in grab_code_blocks(raw, lang=lang):
        add_code_block = True
        for keyword in keyword_ignore:
            if keyword in b:
                add_code_block = False
                break
        if add_code_block:
            all_code = f"{all_code}\n{b}"
    executor(all_code)


def check_md_file(fpath, memory=False, lang="python", keyword_ignore=[]):
    """
    NOTE: copy paste from mktestdocs.__main__ and add the keyword ignore
    Given a markdown file, parse the contents for python code blocks
    and check that each independent block does not cause an error.

    Arguments:
        fpath: path to markdown file
        memory: whether or not previous code-blocks should be remembered
    """
    text = pathlib.Path(fpath).read_text()
    if not memory:
        check_raw_string(text, lang=lang)
    else:
        check_raw_file_full(text, lang=lang, keyword_ignore=keyword_ignore)


files_to_check = [
    *list(pathlib.Path('docs/user_guide').glob('**/*.md')),
    *list(pathlib.Path('docs/data_types').glob('**/*.md')),
]

file_to_remove = []

for file in files_to_check:
    for fn in file_to_skip:
        if fn in str(file):
            file_to_remove.append(file)

for file in file_to_remove:
    files_to_check.remove(file)


@pytest.mark.parametrize('fpath', files_to_check, ids=str)
def test_files_good(fpath):
    check_md_file(fpath=fpath, memory=True, keyword_ignore=['pickle', 'jac'])


def test_readme():
    check_md_file(
        fpath='README.md', memory=True, keyword_ignore=['tensorflow', 'fastapi', 'push', 'langchain']
    )

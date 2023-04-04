import pathlib

import pytest
from mktestdocs import check_md_file


# @pytest.mark.parametrize('fpath', pathlib.Path("docs").glob("**/*.md"), ids=str)
# to use later
@pytest.mark.parametrize(
    'fpath', pathlib.Path('docs/user_guide').glob('**/*.md'), ids=str
)
def test_files_good(fpath):
    check_md_file(fpath=fpath, memory=True)

import random
import string

import pytest


@pytest.fixture(scope='function')
def tmp_index_name():
    letters = string.ascii_lowercase
    random_string = ''.join(random.choice(letters) for _ in range(15))
    return random_string

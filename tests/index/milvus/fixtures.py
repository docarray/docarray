# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import string
import random

import pytest
import time
import os


cur_dir = os.path.dirname(os.path.abspath(__file__))
milvus_yml = os.path.abspath(os.path.join(cur_dir, 'docker-compose.yml'))


@pytest.fixture(scope='session', autouse=True)
def start_storage():
    os.system(f"docker compose -f {milvus_yml} up -d --remove-orphans")
    time.sleep(2)

    yield
    os.system(f"docker compose -f {milvus_yml} down --remove-orphans")


@pytest.fixture(scope='function')
def tmp_index_name():
    letters = string.ascii_lowercase
    random_string = ''.join(random.choice(letters) for _ in range(15))
    return random_string

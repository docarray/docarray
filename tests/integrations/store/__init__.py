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
import tracemalloc
from functools import wraps

from docarray import DocList
from docarray.documents import TextDoc


def get_test_da(n: int):
    return DocList[TextDoc](gen_text_docs(n))


def gen_text_docs(n: int):
    for i in range(n):
        yield TextDoc(text=f'text {i}')


def profile_memory(func):
    """Decorator to profile memory usage of a function.

    Returns:
        original function return value, (current memory usage, peak memory usage)
    """

    @wraps(func)
    def _inner(*args, **kwargs):
        tracemalloc.start()
        ret = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return ret, (current, peak)

    return _inner

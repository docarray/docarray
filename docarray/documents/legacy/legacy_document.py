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
from __future__ import annotations

from typing import Any, Dict, Optional

from docarray import BaseDoc, DocList
from docarray.typing import AnyEmbedding, AnyTensor


class LegacyDocument(BaseDoc):
    """
    This Document is the LegacyDocument. It follows the same schema as in DocArray <=0.21.
    It can be useful to start migrating a codebase from v1 to v2.

    Nevertheless, the API is not totally compatible with DocArray <=0.21 `Document`.
    Indeed, none of the method associated with `Document` are present. Only the schema
    of the data is similar.

    ```python
    from docarray import DocList
    from docarray.documents.legacy import LegacyDocument
    import numpy as np

    doc = LegacyDocument(text='hello')
    doc.url = 'http://myimg.png'
    doc.tensor = np.zeros((3, 224, 224))
    doc.embedding = np.zeros((100, 1))

    doc.tags['price'] = 10

    doc.chunks = DocList[Document]([Document() for _ in range(10)])

    doc.chunks = DocList[Document]([Document() for _ in range(10)])
    ```

    """

    tensor: Optional[AnyTensor] = None
    chunks: Optional[DocList[LegacyDocument]] = None
    matches: Optional[DocList[LegacyDocument]] = None
    blob: Optional[bytes] = None
    text: Optional[str] = None
    url: Optional[str] = None
    embedding: Optional[AnyEmbedding] = None
    tags: Dict[str, Any] = dict()
    scores: Optional[Dict[str, Any]] = None

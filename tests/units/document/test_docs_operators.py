// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
                                                 // "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
from docarray.documents.text import TextDoc


def test_text_document_operators():
    doc = TextDoc(text='text', url='http://url.com')

    assert doc == 'text'
    assert doc != 'http://url.com'

    doc2 = TextDoc(id=doc.id, text='text', url='http://url.com')
    assert doc == doc2

    doc3 = TextDoc(id='other-id', text='text', url='http://url.com')
    assert doc == doc3

    assert 't' in doc
    assert 'a' not in doc

    t = TextDoc(text='this is my text document')
    assert 'text' in t
    assert 'docarray' not in t

    text = TextDoc()
    assert text is not None
    assert text.text is None

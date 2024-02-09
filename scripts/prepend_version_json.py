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
import argparse
import json
from typing import List

parser = argparse.ArgumentParser(prog="Prepender docs/_versions.json")
parser.add_argument(
    "--version",
    type=str,
    help="The version we wish to prepend (e.g. v0.18.0)",
    required=True,
)
args = parser.parse_args()

with open("./docs/_versions.json") as f:
    versions: List[dict] = json.load(f)
    element = {k: v for k, v in args._get_kwargs()}
    if element != versions[0]:
        versions.insert(0, element)

with open("./docs/_versions.json", "w") as f:
    json.dump(versions, f)

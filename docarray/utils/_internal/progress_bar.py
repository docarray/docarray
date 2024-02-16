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
from typing import Optional

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    Text,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class _QPSColumn(TextColumn):
    def render(self, task) -> Text:
        if task.speed:
            _text = f'{task.speed:.0f} QPS'
        else:
            _text = 'unknown'
        if self.markup:
            text = Text.from_markup(_text, style=self.style, justify=self.justify)
        else:
            text = Text(_text, style=self.style, justify=self.justify)
        if self.highlighter:
            self.highlighter.highlight(text)
        return text


def _get_pbar(disable: bool, total: Optional[int] = None):
    columns = (
        SpinnerColumn(),
        TextColumn('[bold]{task.description}'),
        BarColumn(),
        MofNCompleteColumn(),
        '•',
        _QPSColumn('{task.speed} QPS', justify='right', style='progress.data.speed'),
        '•',
        TimeRemainingColumn() if total else TimeElapsedColumn(),
        '•',
        TextColumn(
            '[bold blue]{task.fields[total_size]}',
            justify='right',
            style='progress.filesize',
        ),
    )

    return Progress(
        *columns,
        transient=False,
        disable=disable,
    )


def _get_progressbar(description: str, disable: bool, total: Optional[int]):
    progress = _get_pbar(disable, total)
    task = progress.add_task(description, total=total, start=False, total_size=0)
    return progress, task

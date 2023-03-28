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

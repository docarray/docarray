from rich.progress import (
    Progress,
    BarColumn,
    SpinnerColumn,
    MofNCompleteColumn,
    TextColumn,
    TimeRemainingColumn,
)


def get_progressbar(description, disable, total):
    progress = Progress(
        SpinnerColumn(),
        TextColumn(description),
        BarColumn(),
        MofNCompleteColumn(),
        '•',
        TimeRemainingColumn(),
        '•',
        TextColumn(
            '[bold blue]{task.fields[total_size]}',
            justify='right',
            style='progress.filesize',
        ),
        transient=True,
        disable=disable,
    )
    task = progress.add_task(description, total=total, start=False, total_size=0)
    return progress, task

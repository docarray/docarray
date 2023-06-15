import pydantic


def is_pydantic_v2() -> bool:
    return pydantic.__version__.startswith('2.')


if not is_pydantic_v2():
    from pydantic.validators import bytes_validator

else:

    def bytes_validator(*args, **kwargs):
        raise NotImplementedError('bytes_validator is not implemented in pydantic v2')

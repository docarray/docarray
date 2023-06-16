import pydantic

is_pydantic_v2 = pydantic.__version__.startswith('2.')


if not is_pydantic_v2:
    from pydantic.validators import bytes_validator

else:
    from pydantic.v1.validators import bytes_validator

__all__ = ['is_pydantic_v2', 'bytes_validator']

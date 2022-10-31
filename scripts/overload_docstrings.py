import re


def _get_overload_params_docstring(file_str, tag, class_method=True, indent=' ' * 4):
    method_start_regex = f'# overload_inject_start_{tag}'
    params_start_regex = f':param|EXAMPLE USAGE|.. code-block::'
    docstring_end_regex = f'# overload_inject_end_{tag}|.. # noqa:|:return:'
    # match begin of the method to the end of the docstring (='stub')
    stub_match = re.search(
        rf'({method_start_regex}).*?({docstring_end_regex})', file_str, flags=re.DOTALL
    )
    stub_str = file_str[stub_match.span()[0] : stub_match.span()[1]]
    if tag == 'match':
        print('STUB')
        print(stub_str)
    # match only the params section in the docstring of the stub
    params_match = re.search(
        rf'({params_start_regex}).*?({docstring_end_regex})', stub_str, flags=re.DOTALL
    )
    params_str = stub_str[params_match.span()[0] : params_match.span()[1]]

    #### cleanup ####
    params_str = re.sub('"""', '', params_str, 0, re.DOTALL)  # delete """
    params_str = re.sub(
        rf'{docstring_end_regex}', '', params_str, 1, re.DOTALL
    )  # delete end regex
    params_str = re.sub(
        '\n+$', '', params_str, 1, re.DOTALL
    )  # delete trailing white space
    params_str = re.sub(
        '^\n+', '', params_str, 1, re.DOTALL
    )  # delete leading white space
    params_str = re.sub('\s+$', '', params_str, 1, re.DOTALL)  # delete trailing newline
    params_str = re.sub('^\s+', '', params_str, 1, re.DOTALL)  # delete leading newline
    # add indent back in
    params_str = (
        f'{indent}{indent}{params_str}' if class_method else f'{indent}{params_str}'
    )

    if tag == 'match':
        print('PARAMS')
        print(params_str)

    return params_str


def _get_docstring_title(file_str, tag):
    # extracts the description ('title') of a docstring, i.e. the initial part that has no :param:, :return: etc.
    title_start_regex = f'# implementation_stub_inject_start_{tag}'
    title_end_regex = f':param|EXAMPLE USAGE|.. code-block::|:return:|# implementation_stub_inject_end_{tag}|.. # noqa:'
    doc_str_title_match = re.search(
        rf'({title_start_regex}).*?({title_end_regex})', file_str, flags=re.DOTALL
    )
    doc_str_title = file_str[
        doc_str_title_match.span()[0] : doc_str_title_match.span()[1]
    ]
    # trim of start and end patterns
    doc_str_title = re.sub('"""', '', doc_str_title, 0, re.DOTALL)  # delete """
    doc_str_title = re.sub(
        rf'{title_start_regex}', '', doc_str_title, 1, re.DOTALL
    )  # delete start regex
    doc_str_title = re.sub(
        rf'{title_end_regex}', '', doc_str_title, 1, re.DOTALL
    )  # delete end regex
    doc_str_title = re.sub(
        '\n+$', '', doc_str_title, 1, re.DOTALL
    )  # delete trailing white space
    doc_str_title = re.sub(
        '^\n+', '', doc_str_title, 1, re.DOTALL
    )  # delete leading white space
    doc_str_title = re.sub(
        '\s+$', '', doc_str_title, 1, re.DOTALL
    )  # delete trailing newline
    doc_str_title = re.sub(
        '^\s+', '', doc_str_title, 1, re.DOTALL
    )  # delete leading newline
    return doc_str_title


def fill_implementation_stub(
    doc_str_return,
    return_type,
    filepath,
    overload_fn,
    class_method,
    indent=' ' * 4,
    overload_tags=[],  # from which methods should we gather the docstrings?
    regex_tag=None,
    additional_params=[],  # :param: lines that do not come from the override methods, but from the implementation stub itself
):
    # collects all :param: descriptions from overload methods and adds them to the method stub that has the actual implementation
    file_str = open(filepath).read()
    overload_fn = overload_fn.lower()
    relevant_docstrings = [
        _get_overload_params_docstring(file_str, t, class_method=class_method)
        for t in overload_tags
    ]
    add_param_indent = f'{indent}{indent}' if class_method else f'{indent}'
    relevant_docstrings += [add_param_indent + p for p in additional_params]
    if class_method:
        doc_str = ''
        for i, s in enumerate(relevant_docstrings):
            if i != 0:
                doc_str += '\n'
            doc_str += s
        noqa_str = '\n'.join(
            f'{indent}{indent}.. # noqa: DAR{j}' for j in ['102', '202', '101', '003']
        )
        if return_type:
            return_str = f'\n{indent}{indent}:return: {doc_str_return}'
        else:
            return_str = ''
    else:
        doc_str = ''
        for i, s in enumerate(relevant_docstrings):
            if i != 0:
                doc_str += '\n'
            doc_str += s
        noqa_str = '\n'.join(
            f'{indent}.. # noqa: DAR{j}' for j in ['102', '202', '101', '003']
        )
        if return_type:
            return_str = f'\n{indent}:return: {doc_str_return}'
        else:
            return_str = ''
    if class_method:
        doc_str_title = _get_docstring_title(file_str, regex_tag or overload_fn)
        final_str = f'\n{indent}{indent}"""{doc_str_title}\n\n{doc_str}{return_str}\n\n{noqa_str}\n{indent}{indent}"""'
        final_code = re.sub(
            rf'(# implementation_stub_inject_start_{regex_tag or overload_fn}).*(# implementation_stub_inject_end_{regex_tag or overload_fn}\s)',
            f'\\1\n{indent}{final_str}\n{indent}\\2',
            file_str,
            0,
            re.DOTALL,
        )
    else:
        doc_str_title = _get_docstring_title(file_str, regex_tag or overload_fn)
        final_str = f'\n{indent}"""{doc_str_title}\n\n{doc_str}{return_str}\n\n{noqa_str}\n{indent}"""'
        final_code = re.sub(
            rf'(# implementation_stub_inject_start_{regex_tag or overload_fn}).*(# implementation_stub_inject_end_{regex_tag or overload_fn}\s)',
            f'\\1\n{final_str}\n{indent}\\2',
            file_str,
            0,
            re.DOTALL,
        )

    with open(filepath, 'w') as fp:
        fp.write(final_code)


# param
entries = [
    dict(
        doc_str_return='itself after modification',
        return_type="'T'",
        filepath='../docarray/array/mixins/parallel.py',
        overload_fn='apply_batch',
        class_method=True,  # if it is a method inside class.
        overload_tags=['apply_batch'],
    ),
    dict(
        doc_str_return='itself after modification',
        return_type="'T'",
        filepath='../docarray/array/mixins/parallel.py',
        overload_fn='apply',
        class_method=True,  # if it is a method inside class.
        overload_tags=['apply'],
    ),
    dict(
        doc_str_return='itself after modification',
        return_type="'T'",
        filepath='../docarray/document/mixins/sugar.py',
        overload_fn='match',
        class_method=True,  # if it is a method inside class.
        overload_tags=['match'],
    ),
]


if __name__ == '__main__':
    all_changed_files = set()
    for d in entries:
        fill_implementation_stub(**d)
        all_changed_files.add(d['filepath'])
    for f in all_changed_files:
        print(f)

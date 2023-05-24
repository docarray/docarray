# Contributing to DocArray

Thanks for your interest in contributing to DocArray. We're grateful for your initiative! ❤️

In this guide, we're going to go through the steps for each kind of contribution, and good and bad examples of what to do. We look forward to your contributions!

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [🐞 Bugs and Issues](#-bugs-and-issues)
- [🥇 Making Your First Submission](#-making-your-first-submission)
- [📝 Code style conventions](#-code-style-conventions)
- [☑️ Naming Conventions](#-naming-conventions)
- [➕ Adding a dependency](#adding-a-dependency)
- [💥 Testing DocArray Locally and on CI](#-testing-docarray-locally-and-on-ci)
- [📖 Contributing Documentation](#-contributing-documentation)
- [🙏 Thank You](#-thank-you)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

<a name="-bugs-and-issues"></a>
## 🐞 Bugs and issues

### Submitting issues

We love to get issue reports. But we love it even more if they're in the right format. For any bugs you encounter, we need you to:

* **Describe your problem**: What exactly is the bug. Be as clear and concise as possible
* **Why do you think it's happening?** If you have any insight, here's where to share it

There are also a couple of nice to haves:

* **Environment:** Operating system, DocArray version, python version,...
* **Screenshots:** If they're relevant

<a name="-making-your-first-submission"></a>
## 🥇 Making your first submission

0. Associate your local git config with your GitHub account. If this is your first time using git you can follow [the steps](#associate-with-github-account).
1. Fork the DocArray repo and clone onto your computer. 
1. Configure git pre-commit hooks. Please follow [the steps](#install-pre-commit-hooks)
1. Create a [new branch](#naming-your-branch), for example `fix-docarray-typo-1`.
1. Work on this branch to do the fix/improvement.
1. Commit the changes with the [correct commit style](#writing-your-commit-message).
1. Make a pull request.
1. Submit your pull request and wait for all checks to pass.
1. Request reviews from one of [the code owners](.github/CODEOWNERS).
1. Get a LGTM 👍 and PR gets merged.

**Note:** If you're just fixing a typo or grammatical issue, you can go straight to a pull request.

### Associate with your GitHub account

- Confirm username and email on [your profile page](https://github.com/settings/profile).
- Set git config on your computer.

```shell
git config user.name "YOUR GITHUB NAME"
git config user.email "YOUR GITHUB EMAIL"
```

- (Optional) Reset the commit author if you made commits before you set the git config.

```shell
git checkout YOUR-WORKED-BRANCH
git commit --amend --author="YOUR-GITHUB-NAME <YOUR-GITHUB-EMAIL>" --no-edit
git log  # to confirm the change is effective
git push --force
```

### Installing dependencies using Poetry

We use [Poetry](https://python-poetry.org/) to manage our dependencies.

To get stared with DocArray development you should do:

```shell
pip install poetry
poetry install --all-extras # this will install all of the dependency needed for development
```

This will automatically create a virtual environment and install all the dependency from the `lockfile` of Poetry.

To run your code you need to either activate the environment:

```shell
poetry shell
python XYZ
```
or use `poetry run`:

```shell
poetry run python scratch.py
poetry run pip xyz
poetry run pytest
poetry run XYZ
```

### Install pre-commit hooks

In DocArray we use git's pre-commit hooks in order to make sure the code matches our standards of quality and documentation.
It's easy to configure it:

1. `pip install pre-commit`
1. `pre-commit install`

Now you will be automatically reminded to add docstrings to your code. `black` will take care that your code will match our style. Note that `black` will fail your commit but reformat your code, so you just need to add the files again and commit **again**.

#### Restoring correct git blame

Run `git config blame.ignoreRevsFile .github/.git-blame-ignore-revs`

## 📝 Code style conventions:

Most of our codebase is written in Python. 

### PEP compliance

We comply to the official PEP: E9, F63, F7, F82 code style and required every contribution to follow it. This is enforced by using [ruff](https://github.com/charliermarsh/ruff) in our CI and in our [pre-commit hooks](https://github.com/docarray/docarray/blob/main/CONTRIBUTING.md#install-pre-commit-hooks).

### Python version
DocArray is compatible with Python 3.7 and above, therefore we can't accept contribution that used features from the newest Python versions without ensuring compatibility with python 3.7

### Code formatting

All of our Python codebase follows formatting standard. We are following the [PEP8](https://peps.python.org/pep-0008/) standard, and we require that every code contribution is formatted using [black](https://github.com/psf/black) with the default configurations.
If you have installed the [pre-commit hooks](https://github.com/jina-ai/docarray/blob/main/CONTRIBUTING.md#install-pre-commit-hooks) the formatting should be automatic on every commit. Moreover, our CI will block contributions that do not respect these conventions.

### Type hints

Python is not a strongly typed programming language. Nevertheless, the use of [type hints](https://docs.python.org/3/library/typing.html)  
contributes to a better codebase, especially when reading, reviewing and refactoring. Therefore, we **require every contribution
to use type hints**, unless there are strong reasons for not using them.

Further, DocArray is type checked using **[mypy](https://mypy-lang.org/)**, and all contributions will have to pass this type check.

Note: Example code in the documentation should also follow our code style conventions.


<a name="-naming-conventions"></a>
## ☑️ Naming conventions

For branches, commits, and PRs we follow some basic naming conventions:

* Be descriptive
* Use all lower-case
* Limit punctuation
* Include one of our specified [types](#specify-the-correct-types)
* Short (under 70 characters is best)
* In general, follow the [Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/#summary) guidelines


### Specify the correct types

Type is an important prefix in PR, commit message. For each branch, commit, or PR, we need you to specify the type to help us keep things organized. For example,

```
feat: add hat wobble
^--^  ^------------^
|     |
|     +-> Summary in present tense.
|
+-------> Type: build, ci, chore, docs, feat, fix, refactor, style, or test.
```

- `ci`: Changes to our CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs)
- `docs`: Documentation only changes
- `feat`: A new feature
- `fix`: A bug fix
- `perf`: A code change that improves performance
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests or correcting existing tests
- `chore`: updating grunt tasks etc.; no production code change

### Writing your commit message

A good commit message helps us track DocArray's development. A pull request with a bad commit message will be *rejected* automatically in the CI pipeline.

Commit messages should stick to our [naming conventions](#-naming-conventions) outlined above, and use the format `type(scope?): subject`:

* `type` is one of the [types above](#specify-the-correct-types).
* `scope` is optional, and represents the module your commit is working on.
* `subject` explains the commit, without an ending period`.`

For example, a commit that fixes a bug in the executor module should be phrased as: `fix(executor): fix the bad naming in init function`

> Good examples:
>
```text
fix(elastic): fix batching in elastic document store
feat: add remote api
```

> Bad examples:
>

| Commit message                                                                                  | Feedback                           |
|-------------------------------------------------------------------------------------------------|------------------------------------|
| `doc(101): improved 101 document`                                                               | Should be `docs(101)`              |
| `tests(flow): add unit test to document array`                                                  | Should be `test(array)`            |
| `DOC(101): Improved 101 Documentation`                                                          | All letters should be in lowercase |
| `fix(pea): i fix this issue and this looks really awesome and everything should be working now` | Too long                           |
| `fix(array):fix array serialization`                                                            | Missing space after `:`            |
| `hello: add hello-world`                                                                        | Type `hello` is not allowed        |

#### DCO and signed commit

Commits need to be signed. Indeed, the DocArray repo enforces the [Developer Certificate of Origin](https://developercertificate.org/) via the [DCO](https://github.com/apps/dco) GitHub app.

To sign your commits you need to [use the `-s` argument](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits) when committing:

```
git commit -m -s 'feat: add a new feature'
```

#### What if I mess up?

We all make mistakes. GitHub has a guide on [rewriting commit messages](https://docs.github.com/en/free-pro-team@latest/github/committing-changes-to-your-project/changing-a-commit-message) so they can adhere to our standards.

You can also install [commitlint](https://commitlint.js.org/#/) onto your own machine and check your commit message by running:

```bash
echo "<commit message>" | commitlint
```

### Naming your pull request

We don't enforce naming of PRs and branches, but we recommend you follow the same style. It can simply be one of your commit messages, just copy/paste it, e.g. `fix(readme): improve the readability and move sections`.

<a name="-adding a dependency"></a>
## ➕ Adding a dependency

To add a dependency to DocArray, edit `pyproject.toml` and add your dependency in the `[tool.poetry.dependencies]` section.
Always overwrite poetry default version number (if you used `poetry add XYZ`):
- Pick an appropriate version number. Don't pick the latest version, but rather the oldest that is still compatible.
- Use the `>=` notation instead of `~` to not lock upper limit.

If appropriate, make the dependency optional. For example if it is a new library for a new modality or new vector database.

`mylib = {version = ">=X.y.z", optional = true }`

You will also need to add an extra:

```toml
[tool.poetry.extras]
new_modalities = ['mylib']
```

Note: Manual editing of `pyproject.toml` is equivalent  to `poetry add "mylib>=3.9"  -E new_modalities`

<a name="-testing-docarray-locally-and-on-ci"></a>
## 💥 Testing DocArray Locally and on CI

Locally you can run the tests via:

```bash
poetry install --all-extras
poetry run pip install protobuf==3.19.0
poetry run pip install tensorflow
poetry run pytest -v -s tests
```

For local development we suggest using the following command to run the tests:

```bash
poetry run pytest -v -s tests -m 'not tensorflow and not slow and not internet'
```

This only take a couple of seconds.

### Test policy

Every contribution that adds or modifies the behavior of a feature must include a suite of tests that validates that the feature works as expected.

This allows:

* the reviewer to be very confident that the feature does what it is supposed to do before merging it into the code base.
* the contributors to be sure that they don't break already-merged features when refactoring or modifying the code base.

<a name="-enable-logging"></a>
## Enable logging


If you need to monitor and debug your code, you can enable docarray logging:
```python
import logging

logging.getLogger('docarray').setLevel(logging.DEBUG)
```

<a name="-compiling-protobuf"></a>
## Compiling protobuf

Some changes to the code base require also changing the `.proto` files that describe how DocArray serializes to and from
protobuf messages.

Changes to the `.proto` definitions **should be kept to a minimum**, in order to avoid breaking changes.

If you _do_ make modification in a `.proto` file, you need to **recompile the protobuf definitions.**
In order to maintain compatibility with most of the Python ecosystem, in DocArray we compile to two different protobuf
versions. Therefore, compilation is a two-step process:

### Step 1: Compile using `protoc` version 3.19

1. Download protoc v3.19 as appropriate for your system, e.g. from [here](https://github.com/protocolbuffers/protobuf/releases/tag/v3.19.5)
2. Unzip the file and make `protoc` executable: `chmod +x bin/protoc`
3. Compile the protobuf definitions in the `pb2` directory. From `docarray/proto/` run `path/to/v-3-19/bin/protoc -I . --python_out="pb2" docarray.proto`.

### Step 2: Compile using `protoc` version 3.21

1. Download protoc v3.21 as appropriate for your system, e.g. from [here](https://github.com/protocolbuffers/protobuf/releases/tag/v21.12)
2. Same as above
3. Compile the protobuf definitions in the `pb` directory. From `docarray/proto/` run `path/to/v-3-21/bin/protoc -I . --python_out="pb" docarray.proto`.

<a name="-contributing-documentation"></a>
## 📖 Contributing documentation

Good docs make developers happy, and we love happy developers! We've got a few different types of docs:

* General documentation
* Tutorials/examples
* Docstrings in Python functions in RST format - generated by Sphinx

### Documentation guidelines

1. Decide if your page is a **user guide or a how-to**, like in the `Data Types` section. Make sure it fits its section.
2. Use “**you**” instead of “we” or “I”. It **engages** the reader more.
3. **Sentence case** for headers. (Use [https://convertcase.net/](https://convertcase.net/) to check)
4. Keep sentences short. If possible, **fewer than 13 words**.
5. Only use `backticks` for direct references to code elements.
6. All **acronyms** should be UPPERCASE (Ex. YAML, JSON, HTTP, SSL).
7. Think about the **structure** of the page beforehand. Split it into headers before writing the content.
8. If relevant, include a “**See also**” section at the end.
9. Link to any existing explanations of the concepts you are using.
10. Example code in the documentation should also follow our code style.
11. **Know when to break the rules**. Documentation writing is as much art as it is science. Sometimes you will have to deviate from these rules in order to write good documentation.

### Building documentation on your local machine


#### Steps to build locally

First install the documentation dependency
```
poetry install --with docs
```

Note: if you need to install extra (proto, database, ...) you need to specify those as well.

Then build the documentation:
```bash
cd docs
./makedoc.sh
```

The docs website will be generated in `site`.
To serve it, run:

```bash
cd ..
poetry run mkdocs serve
```

You can now see docs website on [http://localhost:8000](http://localhost:8000) on your browser.
Note: You may have to change the port from 8000 to something else if you already have a server running on that port.

## 🙏 Thank you

Once again, thanks so much for your interest in contributing to DocArray. We're excited to see your contributions!

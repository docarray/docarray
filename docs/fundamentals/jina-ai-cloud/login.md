(login)=
# Authentication

To manage your resources in Jina AI Cloud using DocArray, you need to authenticate to Jina AI Cloud.
Jina AI Cloud offers several ways to login. Read more about [Login & Token Management in Jina AI Cloud](https://docs.jina.ai/jina-ai-cloud/login/).
DocArray also offers convenience methods to login/logout using Python API.

## Login
To login using Python API, use the method {meth}`~docarray.helper.login`:
```python
from docarray import login

login()
```
The method {meth}`~docarray.helper.login` is interactive, meaning that it will prompt you to login in browser. Non-interactive login options are 
available in [this page](https://docs.jina.ai/jina-ai-cloud/login/).

{meth}`~docarray.helper.login` supports notebook environments as well, but it's recommended to use parameter `interactive` 
in that case:
```python
from docarray import login

login(interactive=True)
```
## Logout
To logout, you can use the method {meth}`~docarray.helper.logout`:
```python
from docarray import logout

logout()
```

For more logout methods (CLI), check [this page](https://docs.jina.ai/jina-ai-cloud/login/).
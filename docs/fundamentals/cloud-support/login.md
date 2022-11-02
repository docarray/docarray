(login)=
# Authentication

To manage your resources in Jina AI Cloud using DocArray, you need to authenticate to Jina AI Cloud.
Jina AI Cloud offers several ways to log in. Read more about [Login & Token Management in Jina AI Cloud](https://docs.jina.ai/jina-ai-cloud/login/).
DocArray also offers convenience methods to log in and log out using the Python API.

## Login
To log in using the Python API, use the {meth}`~docarray.helper.login` method:
```python
from docarray import login

login()
```
The {meth}`~docarray.helper.login` method is interactive, meaning that it will prompt you to log in using a browser. Non-interactive login options are 
explained in [Login & Token Management](https://docs.jina.ai/jina-ai-cloud/login/).

{meth}`~docarray.helper.login` supports notebook environments as well, but it's recommended to use parameter `interactive` 
in that case:
```python
from docarray import login

login(interactive=True)
```
## Logout
To log out, you can use the {meth}`~docarray.helper.logout` method:
```python
from docarray import logout

logout()
```

For more logout methods (CLI), see [Login & Token Management](https://docs.jina.ai/jina-ai-cloud/login/).
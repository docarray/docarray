import base64
from typing import Optional

from ...helper import random_identity, download_mermaid_url


class PlotMixin:
    """Provide helper functions for :class:`Document` to plot and visualize itself. """

    @property
    def _mermaid_id(self):
        if not hasattr(self, '__mermaid_id'):
            self.__mermaid_id = random_identity()
        return self.__mermaid_id

    def __mermaid_str__(self):
        results = []
        _id = f'{self._mermaid_id[:3]}~Document~'

        for idx, c in enumerate(self.chunks):
            results.append(
                f'{_id} --> "{idx + 1}/{len(self.chunks)}" {c._mermaid_id[:3]}~Document~: chunks'
            )
            results.append(c.__mermaid_str__())

        for idx, c in enumerate(self.matches):
            results.append(
                f'{_id} ..> "{idx + 1}/{len(self.matches)}" {c._mermaid_id[:3]}~Document~: matches'
            )
            results.append(c.__mermaid_str__())

        content = self.to_dict()
        if 'chunks' in content:
            content.pop('chunks')
        if 'matches' in content:
            content.pop('matches')
        if content:
            results.append(f'class {_id}{{')
            for k, v in content.items():
                if isinstance(v, (str, int, float, bytes)):
                    results.append(f'+{k} {str(v)[:10]}')
                else:
                    results.append(f'+{k}({type(getattr(self, k, v))})')
            results.append('}')

        return '\n'.join(results)

    def _mermaid_to_url(self, img_type: str) -> str:
        """
        Rendering the current flow as a url points to a SVG, it needs internet connection

        :param img_type: the type of image to be generated
        :return: the url pointing to a SVG
        """
        mermaid_str = (
            """
                                                                            %%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#FFC666'}}}%%
                                                                            classDiagram
                        
                                                                                    """
            + self.__mermaid_str__()
        )

        encoded_str = base64.b64encode(bytes(mermaid_str.strip(), 'utf-8')).decode(
            'utf-8'
        )

        return f'https://mermaid.ink/{img_type}/{encoded_str}'

    def _ipython_display_(self):
        """Displays the object in IPython as a side effect"""
        self.plot(inline_display=True)

    def plot(self, output: Optional[str] = None, inline_display: bool = False) -> None:
        """
        Visualize the Document recursively.

        :param output: a filename specifying the name of the image to be created,
                    the suffix svg/jpg determines the file type of the output image
        :param inline_display: show image directly inside the Jupyter Notebook
        """
        image_type = 'svg'
        if (
            not output.endswith('.svg')
            and not output.endswith('.jpg')
            and not output.endswith('.jpeg')
        ):
            raise ValueError('`output` can be only SVG/JPG format')
        elif output.endswith('.jpg') or output.endswith('.jpeg'):
            image_type = 'img'

        url = self._mermaid_to_url(image_type)
        showed = False
        if inline_display:
            try:
                from IPython.display import Image, display

                display(Image(url=url))
                showed = True
            except:
                # no need to panic users
                pass

        if output:
            download_mermaid_url(url, output)
        elif not showed:
            print(f'Document visualization: {url}')

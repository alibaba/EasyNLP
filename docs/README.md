### How to maintain docs

### 1. Install easy transfer

```
$ cd /to/dir/EasyNLP
$ python setup.py install
```

### 2. Install sphinx

```bash
$ pip install sphinx
$ pip install sphinx_rtd_theme
```

### 3. Add modules

#### 3.1  Add class or functions in existing files

You need to add class or functions with `docstring` into the attached file.

1. Google Python Style Guide [link](http://google.github.io/styleguide/pyguide.html#381-docstrings)
1. Google docstring Sample [link](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
1. Sample project：torch.nn.modules.conv [link](https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv1d)
1. Take `easynlp.appzoo.classification.BertTextClassify` as example：

````python
class BertTextClassify(BertPreTrainedModel):
    """
        Transformer model from ```Attention Is All You Need''',
        Original paper: https://arxiv.org/abs/1706.03762

        Args:
            num_token (int): vocab size.
            num_layer (int): num of layer.
            num_head (int): num of attention heads.
            embedding_dim (int): embedding dimension.
            attention_head_dim (int): attention head dimension.
            feed_forward_dim (int): feed forward dimension.
            initializer: initializer type.
            activation: activation function.
            dropout (float): dropout rate (0.0 to 1.0).
            attention_dropout (float): dropout rate for attention layer.

        Returns: None
    """
````

#### 3.2  Add new file

For example, if you need to add a new file in `easynlp/data` and the file name is `blackmagic.py` with a `BlackMagic` class:

1. Add `docstring` to the code
1. In `docs/source/api/data.rst`，Find a position for `blackmagic` and add

```rst
blackmagic
--------------------------------------

.. automodule:: easynlp.data.blackmagic
    :members:
    :undoc-members:
    :show-inheritance:

```

#### 3.3  Add new directory

For example, you want to add a `magic` directory in `ez_transfer`，and there is a file named `blackmagic.py` with a `BlackMagic` class:

1. Add `docstring` to the code
1. Add file `docs/source/api/magic.rst` and write the following line

```rst
ez\_transfer.magic
===========================
```

3. In `docs/source/api/magic.rst`，Find a position for `blackmagic` and add

```rst
blackmagic
--------------------------------------

.. automodule:: ez_transfer.layers.blackmagic
    :members:
    :undoc-members:
    :show-inheritance:

```

### 4.  Generate doc html

```bash
$ cd docs/
$ sh build_docs.sh
```

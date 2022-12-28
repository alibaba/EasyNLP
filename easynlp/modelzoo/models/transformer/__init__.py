from typing import TYPE_CHECKING

from ...file_utils import (
    _BaseLazyModule,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_transformer": ["TransformerConfig"],
    "tokenization_transformer": ["TransformerTokenizer"],
}

if is_torch_available():
    _import_structure["modeling_transformer"] = [
        "TransformerModel"
    ]

if TYPE_CHECKING:
    # from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
    from .configuration_transformer import TransformerConfig
    from .tokenization_transformer import TransformerTokenizer

    if is_torch_available():
        from .modeling_transformer import (
            # BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TransformerModel
            # load_tf_weights_in_bert,
        )

else:
    import importlib
    import os
    import sys

    class _LazyModule(_BaseLazyModule):
        """
        Module class that surfaces all objects but only performs associated imports when the objects are requested.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        def _get_module(self, module_name: str):
            return importlib.import_module("." + module_name, self.__name__)

    sys.modules[__name__] = _LazyModule(__name__, _import_structure)

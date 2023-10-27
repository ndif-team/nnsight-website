nnsight.LanguageModel
=====================

.. py:class:: LanguageModel(*args, **kwargs)

    A wrapper class that encapsulates transformer-based language models for causal language modeling. It provides an interface for loading both meta and local models, preparing inputs, and performing inference and text generation.

    Inherits from :py:class:AbstractModel.

    :param args: Variable length argument list.
    :param kwargs: Arbitrary keyword arguments.

.. py:method:: init(self, *args, **kwargs) -> None

    Initializes the LanguageModel instance. Sets up the configuration, tokenizer, meta_model, and local_model as None before calling the superclass's initializer.

Example
"""""""

.. code-block:: python

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LanguageModel('gpt2',device_map=device)
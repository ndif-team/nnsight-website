nnsight.toolbox.optim.LORA
==========================

.. py:class:: LORA(module: Module, r: int) -> None

    The Low-Rank Adaptation (LORA) optimizer, implemented as a subclass of :py:class:`Optimization`. 
    LORA is designed to inject trainable low-rank matrices into the model, facilitating efficient 
    adaptation of large language models to specific tasks without extensive fine-tuning.

    :param module: The module to which LORA is being applied.
    :type module: Module
    :param r: The rank for the decomposition matrices.
    :type r: int

    .. py:attribute:: module

        The module to which LORA is being applied.

    .. py:attribute:: r

        The rank for the decomposition matrices.

    .. py:attribute:: WA

        The first trainable low-rank matrix of shape (input_dimension, r), where input_dimension 
        is derived from the input shape of the module.

    .. py:attribute:: WB

        The second trainable low-rank matrix of shape (r, output_dimension), where output_dimension 
        is derived from the output shape of the module.

.. py:method:: LORA.__call__(alpha: float = 1.0) -> Any

    Applies the LORA transformation to the input of the module, adjusting the output of the module accordingly.

    :param alpha: A scaling factor for the combined output. Default is 1.0.
    :type alpha: float
    :return: None

.. py:method:: LORA.parameters() -> list

    Provides a list of the trainable parameters managed by the LORA optimizer.

    :return: A list containing the trainable low-rank matrices WA and WB.
    :rtype: list


Usage Example
-------------

.. code-block:: python

    from typing import Any
    import torch
    from nnsight import AbstractModel, LanguageModel, util
    from nnsight.Module import Module
    from nnsight.toolbox.optim.lora import LORA
    from torch.utils.data import DataLoader, Dataset

    # ... rest of your code

    lora = LORA(model.transformer.h[0].mlp, 10)
    optimizer = torch.optim.AdamW(lora.parameters(), lr=.1)

    # ... rest of your code

    with model.generate() as generator:
        with generator.invoke(dataset[0][0]) as invoker:
            lora()

    print(model.tokenizer.decode(generator.output[0]))


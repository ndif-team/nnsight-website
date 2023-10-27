Forward
=======

Operations within a forward context interact with the model's forward pass. 

.. py:function:: nnsight.models.AbstractModel.forward(inputs: Union[torch.tensor, List], *args, **kwargs) -> Runner

    Function implemented by all nnsight models that returns an instance of the :py:class:`nnsight.contexts.Runner` class.

    :param Union[torch.tensor, List] input: Batch or sequence of tokens in string or tensor format. 


.. py:class:: Runner(model: "AbstractModel", input: Union[torch.tensor, List], *args, inference=False, **kwargs) -> None

    The Runner class orchestrates the execution of the model either for inference or training. 
    It sets up an execution graph and manages inputs, outputs, and interventions during the model run.

    :param model: The model object this runner is for.
    :type model: AbstractModel
    :param Union[torch.tensor, List] input: The input data for the model.
    :param args: Additional arguments for the model.
    :param inference: Whether the model is being run in inference mode. Default is False.
    :type inference: bool
    :param kwargs: Additional keyword arguments for the model.

    .. py:attribute:: model

        The model object this runner is for.

    .. py:attribute:: input

        The input data for the model.

    .. py:attribute:: inference

        Whether the model is being run in inference mode.

    .. py:attribute:: args

        Additional arguments for the model.

    .. py:attribute:: kwargs

        Additional keyword arguments for the model.

    .. py:attribute:: graph

        The execution graph for managing interventions during the model run.

    .. py:attribute:: batch_size

        The current size of the batch being processed.

    .. py:attribute:: prompts

        A list of prompts used by the runner.

    .. py:attribute:: generation_idx

        The current generation index, useful for managing interventions.

    .. py:attribute:: output

        The output of the model after execution.

    .. py:method:: __enter__() -> Runner

        Prepares the runner for a model execution session.

    .. py:method:: __exit__(exc_type, exc_val, exc_tb) -> None

        Cleans up after a model execution session and runs the model with the provided inputs and arguments.

Examples
--------

.. code-block:: python

    # Other imports
    from nnsight.toolbox.optim.lora import LORA
    from torch.utils.data import DataLoader, Dataset

    # ... rest of your code

    for epoch in range(epochs):

        for i, (inputs, targets) in enumerate(dataloader):

            optimizer.zero_grad()
            with model.forward(inputs, inference=False) as runner:
                lora()
                logits = model.lm_head.output.save()

            loss = lossfn(logits.value[:, -1], targets)
            
            loss.backward()
            optimizer.step()

    # ... rest of your code


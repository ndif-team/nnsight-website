Invoker
=======

Within the generation context, we create invocation contexts to specify the actual prompts we want to run. Inside of the block, all operations/interventions will be applied to the processing of the specified prompt. 


.. py:function:: nnsight.contexts.Generator.invoke(input: Union[torch.tensor, List], *args, **kwargs) -> Invoker

    Returns a :py:class:`nnsight.contexts.Invoker` context.

    :param Union[torch.tensor, List] input: A sequence or batch of prompts.

.. py:class:: nnsight.contexts.Invoker(generator: "Generator", input: Union[torch.tensor, List], *args, **kwargs) -> None

    The Invoker class is responsible for setting up the environment for a model generation invocation.
    It sets up input, arguments, and keyword arguments for the model generation, and also keeps track
    of tokenized input and generated token ids.

    :param generator: The Generator object that orchestrates the model generation.
    :type generator: Generator
    :param input: The input data for the model.
    :param args: Additional arguments for the model.
    :param kwargs: Additional keyword arguments for the model.

    .. py:method:: __enter__() -> Invoker

        Sets up the environment for a new model generation invocation.

    .. py:method:: __exit__(exc_type, exc_val, exc_tb) -> None

        Cleans up the environment after a model generation invocation.

    .. py:method:: next() -> None

        Prepares for the next generation iteration by incrementing the generation index.

    .. py:method:: save_all() -> Dict[str, Proxy]

        Saves the output of all model modules and returns a dictionary mapping module paths to save proxies.


Examples
--------

.. code-block:: python

    with model.generate(max_new_tokens=3) as generator:
        
        with generator.invoke("Madison square garden is located in the city of New") as invoker:

            lm_head_one = model.lm_head.output.save()

        with generator.invoke("The Eiffel Tower is in the city of") as invoker:

            lm_head_two = model.lm_head.output.save()

Multiple invocation contexts can be called within a generate context. Operations within different contexts have no impact on other contexts unless explicitly cross applied. See :doc:`cross prompt intervention </features/crossprompt>` for more.




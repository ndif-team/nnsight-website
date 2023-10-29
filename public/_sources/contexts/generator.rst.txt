Generator
=========

The generation context creates a graph of operations which are carried out on the actual model after the block is exited. 

.. py:function:: nnsight.model.AbstractModel.generate(*args, **kwargs) -> Generator

    Returns a :py:class:`nnsight.contexts.Generator` context for a model.

    :param kwargs: Keyword arguments are passed downstream to `AutoModelForCausalLM.generate(...) <https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate>`_. See linked docs for reference.


.. py:class:: nnsight.contexts.Generator(model: AbstractModel, *args, blocking: bool = True, server: bool = False, **kwargs) -> None

    The Generator class orchestrates model generation, either locally or on a server, 
    and facilitates user interventions during the generation process.

    :param model: The model object this generator is for.
    :type model: AbstractModel
    :param args: Arguments for calling the model.
    :param blocking: Whether to block and wait for responses when using device_map='server'.
    :type blocking: bool
    :param server: Whether to run generation on a server.
    :type server: bool
    :param kwargs: Keyword arguments for calling the model.

    .. py:function:: __enter__() -> Generator

        Prepares the generator for a model generation session.

    .. py:function:: __exit__(exc_type, exc_val, exc_tb) -> None

        Cleans up after a model generation session, running the model either locally or on a server.

    .. py:function:: run_local()

        Runs the model locally and stores the output.

    .. py:function:: run_server()

        Prepares and sends a request to a server to run the model.

    .. py:function:: blocking_request(request: pydantics.RequestModel)

        Sends a blocking request to a server to run the model, waiting for a response.

    .. py:function:: non_blocking_request(request: pydantics.RequestModel)

        Sends a non-blocking request to a server to run the model.

    .. py:function:: invoke(input, *args, **kwargs) -> Invoker
        :noindex:

        Creates and returns an Invoker object for a model generation invocation.


Examples
--------

.. code-block:: python

    with model.generate(max_new_tokens=1) as generator:
        with generator.invoke(gpt2_tokens) as invoker:
            hidden_states = model.transformer.h[-1].output[0].save()

We create a generation context block by calling ``.generate(...)`` on the model object. This denotes we wish to actually generate tokens given some prompts.

Now calling ``.generate(...)`` does not actually initialize or run the model. Only after the with generator block is exited, is the acually model loaded and ran. All operations in the block are "proxies" which essentially creates a graph of operations we wish to carry out later.



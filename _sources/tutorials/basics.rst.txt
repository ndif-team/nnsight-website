Basics
======

This notebook covers the basic features of the Engine API. If you'd like to follow along, check out the `Colab <https://colab.research.google.com/drive/1A3Q7jDIjtyPCaY842c27fEhCWrZFVM5E?usp=sharing>`_.

* :ref:`loading-models`
* :ref:`accessing_activations`
* :ref:`intervening`

.. _loading-models:

Loading Models
--------------

With nnsight, you can access open source LLMs by referencing a Hugging Face repo ID.
For this demo we'll look at `GPT-2 Small <https://huggingface.co/gpt2>`_, an 80M parameter model.

.. code-block:: python

    # Declare the model and load onto device
    model = LanguageModel('gpt2',device_map=device)

Generate some text.

.. code-block:: python

    prompt = 'The Eiffel Tower is in the city of'

    with model.generate(max_new_tokens=1) as generator:
        with generator.invoke(prompt) as invoker:
            hidden_states = model.transformer.h[-1].output[0].save()

    output = generator.output

    print([model.tokenizer.decode(t) for t in output])
    >>> ['The Eiffel Tower is in the city of Paris']

Let's break this down piece by piece.

**We create a generation context block** by calling ``.generate(...)`` on the model object. This denotes that we wish to generate tokens given some prompts.

.. code-block:: python

    with model.generate(max_new_tokens=1) as generator:

Calling ``.generate(...)`` does not actually initialize or run the model. Only after the block is exited is the model actually loaded and run. All operations in the block are proxies which create a graph of operations to be carried out later. 

**Within the generation context,** we create invocation contexts to specify the actual prompts we want to run.

.. code-block:: python

    with generator.invoke(prompt) as invoker:

**Within an invoke context**, all operations/interventions will be applied to the processing of the prompt. Models can be run on a variety of input formats. See model inputs for more.

Finally, we can access raw tensors and activations at any point in the model. *But what can we do with these activations?*

.. _accessing_activations:

Accessing Activations
---------------------

The first basic feature is to break open the black box and look at all of the internal activations of the model. 

.. code-block:: python
    :emphasize-lines: 6

    gpt2_text = "Natural language processing tasks, such as..."
    gpt2_tokens = model.tokenizer.encode(gpt2_text)

    with model.generate(max_new_tokens=1) as generator:
        with generator.invoke(gpt2_tokens) as invoker:
            hidden_states = model.transformer.h[-1].output[0].save()

Lets focus on the highlighted line.

**First,** ``model.transformer.h[-1]`` accesses a module in the computation graph, specifically the last transformer layer ``h[-1]``. 

**Then,** ``.output`` returns a proxy for the output of this module. In other words, when we get to the output of this module during inference, grab it and perform any operations we define on it. The outputs become operational proxies, one for getting the 0th index of the output, and another for saving the output. 

.. code-block:: python

    print(model.transformer.h[-1].output.shape)
    >>> (torch.Size([1, 10, 768]), \
        (torch.Size([1, 12, 10, 64]), torch.Size([1, 12, 10, 64])))

We take the 0th index because the output of gpt2 transformer layers are a *tuple* where the first index is the actual hidden states and the last two are from attention. 

    **Other Operations on Proxies** 

    ``.shape`` can be called on any proxy to get what shape the value will eventually be.
    
    ``.input`` similarly returns a proxy for the inputs to this module. 

**Finally,** ``.save()`` informs the computation graph to clone the value of a proxy, allowing us to access the value of a proxy after generation. During processing of the intervention computational graph, when the value of a proxy is no longer ever needed, its value is dereferenced and destroyed.

After exiting the generator context, the model is ran with the specified arguments and intervention graph. ``generator.output`` is populated with the actual output and ``hidden_states.value`` will contain the value.

.. code-block:: python

    output = generator.output
    hidden_states = hidden_states.value

    print(output)
    >>> tensor([[35364,  3303,  7587,  8861,    11,   884,   355,  1808, 18877, 11,
          4572, 11059,    11,  3555, 35915,    11,   290, 15676,  1634,    11,
           389,  6032, 10448,   351, 28679,  4673,   319,  8861,   431,  7790,
         40522,    13,  2102]], device='cuda:0')

    print(hidden_states)
    >>> tensor([[[ -0.2059,   0.1688,  -2.0503,  ...,  -0.3703,  -0.2015,  -1.6594],
            [ -3.9412,  -0.2137,  -8.5667,  ...,   6.3562,   4.1276,   3.6006],
            [ -2.0798,  -1.5781,  -6.1944,  ...,   4.8023,   5.6864,  -2.6289],
            ...,
            [ -2.1180,  -6.4320, -20.7147,  ...,   8.7145,   2.3738,   3.4004],
            [ -1.1358,  -3.9569, -20.3060,  ...,   7.1600,   1.6868,   0.9850],
            [ -1.7206,  -4.7800,  -1.1185,  ...,   3.1680,   3.7024,   0.2865]]],
        device='cuda:0')

.. _intervening:

Intervening on Activations
--------------------------

The key features here are **operation** and **setting**. Within an invoke context, most basic operations and torch operations work on proxies and are added to the computation graph. We can also use the assignment ``=`` operator to edit and intervene on the flow of information.

As a basic example, let's `ablate <https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=fh-HJyz1CgUVrXuoiban6bYx>`_ head 7 in layer 0 on the text above. 

.. code-block:: python

    layer_to_ablate = 0
    head_index_to_ablate = 7

    with model.generate(max_new_tokens=1) as generator:
        with generator.invoke(gpt2_tokens) as invoker:
            normal_lm_head = model.lm_head.output.save()
            
        with generator.invoke(gpt2_tokens) as invoker:
            attention_pattern = model.transformer.h[layer_to_ablate].attn.value.output
            attention_pattern[:,head_index_to_ablate,:,:] = 0. 
            ablated_lm_head = model.lm_head.output.save()

    normal_lm_head = normal_lm_head.value
    ablated_lm_head = ablated_lm_head.value

As a result of ablating the head, we observe a noticable change in loss. 

.. code-block:: python

    tensor_tokens = torch.tensor([gpt2_tokens]).to(device)
    print(cross_entropy_loss(normal_lm_head, tensor_tokens, shift=True))
    >>> tensor(4.0187)

    print(cross_entropy_loss(ablated_lm_head, tensor_tokens, shift=True))
    >>> tensor(4.2913)


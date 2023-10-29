Cross Prompt intervention
=========================

Intervention operations work cross prompt! Use two invocations within the same generation block and operations can work between them.

In this case, we grab the token embeddings coming from the first prompt, ``"Madison square garden is located in the city of New"`` and replace the embeddings of the second prompt with them.

.. jupyter-input::

    prompt = "Madison square garden is located in the city of New"

    with model.generate(max_new_tokens=3) as generator:

    with generator.invoke(prompt) as invoker:

        embeddings = model.transformer.wte.output

    with generator.invoke("_ _ _ _ _ _ _ _ _ _") as invoker:

        model.transformer.wte.output = embeddings

    print(model.tokenizer.decode(generator.output[0]))
    print(model.tokenizer.decode(generator.output[1]))

.. jupyter-output::

    Madison square garden is located in the city of New York City.
    _ _ _ _ _ _ _ _ _ _ York City.


We also could have entered a pre-saved embedding tensor as shown here:

.. jupyter-input::

    prompt = "Madison square garden is located in the city of New"

    with model.generate(max_new_tokens=3) as generator:

    with generator.invoke(prompt) as invoker:

        embeddings = model.transformer.wte.output.save()

    print(model.tokenizer.decode(generator.output[0]))
    print(embeddings.value)

    with model.generate(max_new_tokens=3) as generator:

        with generator.invoke("_ _ _ _ _ _ _ _ _ _") as invoker:

            model.transformer.wte.output = embeddings.value

    print(model.tokenizer.decode(generator.output[0]))

.. jupyter-output::

    Madison square garden is located in the city of New York City.

    tensor([[[-0.0063, -0.1795,  0.2060,  ...,  0.1978,  0.1870, -0.1588],
            [-0.1256, -0.0201,  0.0853,  ...,  0.0117, -0.1322, -0.0054],
            [ 0.1283, -0.1087,  0.1027,  ..., -0.0407, -0.1525, -0.1283],
            ...,
            [ 0.0030,  0.0874,  0.1263,  ..., -0.0033,  0.1747, -0.0309],
            [-0.0572,  0.0183,  0.0333,  ..., -0.0689, -0.0931, -0.0714],
            [-0.1341, -0.0436,  0.2236,  ..., -0.1965,  0.0693, -0.1080]]],
        device='cuda:0')

    _ _ _ _ _ _ _ _ _ _ York City.

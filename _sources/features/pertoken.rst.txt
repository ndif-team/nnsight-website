Per Token Indexing
==================

When indexing hidden states for specific tokens, use ``.token[<idx>]`` or ``.t[<idx>]``. This is because if there are multiple invocations, padding is performed on the left side so these helper functions index from the back.

Here we just get the hidden states of the first token:

.. code-block:: python
    
    with model.generate(max_new_tokens=1) as generator:
    with generator.invoke('The Eiffel Tower is in the city of') as invoker:

        hidden_states = model.transformer.h[-1].output[0].t[0].save()

    output = generator.output
    hidden_states = hidden_states.value
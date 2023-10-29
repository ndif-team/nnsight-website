Multiple Token Generation
=========================


When generating more than one token, use ``invoker.next()`` to denote following interventions should be applied to the subsequent generations.

Here we again generate using gpt2, but generate three tokens and save the hidden states of the last layer for each one:

.. code-block:: python

    with model.generate(max_new_tokens=3) as generator:
        with generator.invoke('The Eiffel Tower is in the city of') as invoker:

            hidden_states1 = model.transformer.h[-1].output[0].save()

            invoker.next()

            hidden_states2 = model.transformer.h[-1].output[0].save()

            invoker.next()

            hidden_states3 = model.transformer.h[-1].output[0].save()


    output = generator.output
    hidden_states1 = hidden_states1.value
    hidden_states2 = hidden_states2.value
    hidden_states3 = hidden_states3.value
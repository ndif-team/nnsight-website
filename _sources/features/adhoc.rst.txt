Ad-Hoc Module
=============

You can apply modules in the model's module tree at any point during computation, even if it's out of order.

Here we get the hidden states of the last layer like usual. We also chain apply ``model.transformer.ln_f`` and ``model.lm_head`` in order to "decode" the hidden states into vocabularly space. Applying softmax and then argmax allows us to then transform the vocabulary space hidden states into actually tokens which we can then use the tokenizer to decode.

.. jupyter-input::

    with model.generate() as generator:
    with generator.invoke('The Eiffel Tower is in the city of') as invoker:

        hidden_states = model.transformer.h[-1].output[0]
        hidden_states = model.lm_head(model.transformer.ln_f(hidden_states)).save()
        tokens = torch.softmax(hidden_states, dim=2).argmax(dim=2).save()

    print(hidden_states.value)
    print(tokens.value)
    print(model.tokenizer.decode(tokens.value[0]))

.. jupyter-output::

    tensor([[[ -36.2874,  -35.0114,  -38.0793,  ...,  -40.5163,  -41.3759,
           -34.9193],
         [ -68.8886,  -70.1562,  -71.8408,  ...,  -80.4195,  -78.2552,
           -71.1206],
         [ -82.2950,  -81.6519,  -83.9941,  ...,  -94.4878,  -94.5194,
           -85.6998],
         ...,
         [-113.8675, -111.8628, -113.6634,  ..., -116.7652, -114.8267,
          -112.3621],
         [ -81.8531,  -83.3006,  -91.8192,  ...,  -92.9943,  -89.8382,
           -85.6898],
         [-103.9307, -102.5054, -105.1563,  ..., -109.3099, -110.4195,
          -103.1395]]], device='cuda:0')

    tensor([[ 198,   12,  417, 8765,  318,  257,  262, 3504, 7372, 6342]],
        device='cuda:0')
    
    -el Tower is a the middle centre Paris
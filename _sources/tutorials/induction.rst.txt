Searching for Induction Heads
=============================

Let's look for `induction heads <https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html>`_ in GPT-2 Small. 

Induction circuits are a very important circuit in generative language models, which are used to detect and continue repeated subsequences. They consist of two heads in separate layers that compose together, a **previous token head** which always attends to the previous token, and an **induction head** which attends to the token *after* an earlier copy of the current token. 

To see why this is important, let's say that the model is trying to predict the next token in a news article about Michael Jordan. The token " Michael", in general, could be followed by many surnames. But an induction head will look from that occurence of " Michael" to the token after previous occurences of " Michael", ie " Jordan" and can confidently predict that that will come next.

An interesting fact about induction heads is that they generalise to arbitrary sequences of repeated tokens. We can see this by generating sequences of 50 random tokens, repeated twice, and plotting the average loss at predicting the next token, by position. We see that the model goes from terrible to very good at the halfway point.

.. code-block:: python

    # Declare the model and load onto device
    model = LanguageModel('gpt2',device_map=device)

    batch_size = 10
    seq_len = 50
    random_tokens = torch.randint(1000, 10000, (batch_size, seq_len))
    repeated_tokens = einops.repeat(random_tokens, "batch seq_len -> batch (2 seq_len)")

    with model.generate(max_new_tokens=1) as generator:
        with generator.invoke(repeated_tokens) as invoker:
            token_ids = invoker.ids
            logits = model.lm_head.output.save()

    loss = cross_entropy_loss(logits.value, token_ids, shift=True, avg_token=False)
    line(loss, xaxis="Position", yaxis="Loss", title="Loss by position on random repeated tokens")

.. chart:: charts/line.json

    Loss by position on random repeated tokens

The induction heads will be attending from the second occurence of each token to the token *after* its first occurence, ie the token ``50-1==49`` places back. So by looking at the average attention paid 49 tokens back, we can identify induction heads!

.. code-block:: python

    with model.generate(max_new_tokens=1, output_attentions=True) as generator:

        with generator.invoke(repeated_tokens, output_attentions=True) as invoker:

            attn_hidden_states = [model.transformer.h[layer_idx].attn.output[2].save() for layer_idx in range(len(model.transformer.h))]

    attn_hidden_states = torch.stack([hs.value for hs in attn_hidden_states]).diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
    induction_score = attn_hidden_states.mean(1).mean(-1).cpu()

    imshow(induction_score, xaxis="Head", yaxis="Layer", title="Induction Score by Head")

.. chart:: charts/induction.json

    Induction Score by Head
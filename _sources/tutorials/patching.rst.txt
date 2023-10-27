Patching
========

Let's apply `activation patching <https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=qeWBvs-R-taFfcCq-S_hgMqx>`_ on the `Indirect Object Identification <https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=iWsV3s5Kdd2ca3zNgXr5UPHa>`_ (IOI) task. 

The IOI task is the task of identifying that a sentence like "After John and Mary went to the store, Mary gave a bottle of milk to" continues with " John" rather than " Mary" (ie, finding the indirect object), and Redwood Research have `an excellent paper studying the underlying circuit in GPT-2 Small <https://arxiv.org/abs/2211.00593>`_.

`Activation patching <https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=qeWBvs-R-taFfcCq-S_hgMqx>`_ is a technique from `Kevin Meng and David Bau's excellent ROME paper <https://rome.baulab.info/>`_. The goal is to identify which model activations are important for completing a task. We do this by setting up a **clean prompt** and a **corrupted prompt** and a **metric** for performance on the task. We then pick a specific model activation, run the model on the corrupted prompt, but then *intervene* on that activation and patch in its value when run on the clean prompt. We then apply the metric, and see how much this patch has recovered the clean performance. 

.. code-block:: python

    clean_prompt = \
    "After John and Mary went to the store, Mary gave a bottle of milk to"
    corrupted_prompt = \
    "After John and Mary went to the store, John gave a bottle of milk to"

Here, our clean prompt is "After John and Mary went to the store, **Mary** gave a bottle of milk to", our corrupted prompt is "After John and Mary went to the store, **John** gave a bottle of milk to", and our metric is the difference between the correct logit (John) and the incorrect logit (Mary) on the final token. 

.. code-block:: python

    # Declare the model and load onto device
    model = LanguageModel('gpt2',device_map=device)

    clean_tokens = model.tokenizer.encode(clean_prompt)
    corrupted_tokens = model.tokenizer.encode(corrupted_prompt)

    def logits_to_logit_diff(logits, correct_answer=" John", incorrect_answer=" Mary"):
        correct_index = model.tokenizer.encode(correct_answer)
        incorrect_index = model.tokenizer.encode(incorrect_answer)
        return logits[0, -1, correct_index] - logits[0, -1, incorrect_index]

    clean_resid = []

    with model.generate(max_new_tokens=1) as generator:
        with generator.invoke(clean_tokens) as invoker:
            clean_logits = model.lm_head.output.save()
            for layer in range(12):
                clean_resid.append(model.transformer.h[layer].input.save())

        with generator.invoke(corrupted_tokens) as invoker:
            corrupted_logits = model.lm_head.output.save()

Note how we created a python list structure to store the residual stream input to each layer. Because we're accessing activations through **intervention proxies** within the generate context, we can't save these objects to a tensor or numpy array.

Below, we see that the logit difference is significantly positive on the clean prompt, and significantly negative on the corrupted prompt, showing that the model is capable of doing the task!

.. code-block:: python

    clean_logit_diff = logits_to_logit_diff(clean_logits.value)
    corrupted_logit_diff = logits_to_logit_diff(corrupted_logits.value)

    print(clean_logit_diff)
    >>> tensor([4.1235], device='cuda:0')

    print(corrupted_logit_diff)
    >>> tensor([-2.2725], device='cuda:0')

Here, we'll patch in the residual stream at the start of a specific layer and at a specific position. This will let us see how much the model is using the residual stream at that layer and position to represent the key information for the task.

Note the design consideration made with respect to how the engine API performs interventions. 

We augment the tokens to a batch, shape ``[seq_len, seq_len]``. By passing a batch of duplicate prompts into the model, the shape of the residual stream is now ``[batch seq_len d_model]`` rather than ``[1 seq_len d_model]``. Within the context, we loop through the ``batch`` dimension and patch each respective position in the sequence. 

*Compare* this to a nested for loop that calls the generate context ``layer × seq_len`` times. Because the model is only run after the generate context is exited, a nested for loop would run 180 model calls versus 12.

.. code-block:: python

    seq_len = len(corrupted_tokens)
    n_layers = 12
    results = []
    corrupted_tokens_list = [corrupted_tokens for i in range(seq_len)]

    for layer in tqdm(range(n_layers)):
        with model.generate(max_new_tokens=1) as generator:
            with generator.invoke(corrupted_tokens_list) as invoker:
                resid_pre = model.transformer.h[layer].input[0]
                # Iterate through the batch dimension
                for pos in range(seq_len):
                    # Get saved clean residuals
                    clean_resid_pre = clean_resid[layer].value[0][:,pos,:]
                    resid_pre[pos, pos, :] = clean_resid_pre
        
                patched_logits = model.lm_head.output
                results.append(patched_logits.save())

We can now visualize the results, and see that this computation is extremely localised within the model. Initially, the second subject (Mary) token is all that matters (naturally, as it's the only different token), and all relevant information remains here until heads in layer 7 and 8 move this to the final token where it's used to predict the indirect object.

(Note - the heads are in layer 7 and 8, not 8 and 9, because we patched in the residual stream at the *start* of each layer)

.. code-block:: python
    
    patched_results = np.zeros((12,16))

    def patching_result(patched_logit_diff):
        return (patched_logit_diff - corrupted_logit_diff)/(clean_logit_diff - corrupted_logit_diff)

    for layer in range(12):
        for pos in range(16):
            patched_results[layer, pos] = patching_result(logits_to_logit_diff(results[layer].value[pos].unsqueeze(0)))

    # Add the index to the end of the label, because plotly doesn't like duplicate labels
    token_labels = [f"{token}_{index}" for index, token in enumerate([model.tokenizer.decode(t) for t in clean_tokens])]
    temp_plot = imshow(patched_results, x=token_labels, xaxis="Position", yaxis="Layer", title="Normalized Logit Difference After Patching Residual Stream on the IOI Task")

.. chart:: charts/chart_schema.json

    Loss by position on random repeated tokens
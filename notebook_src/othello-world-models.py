# %% [markdown]
# <details>
# <summary><b>Info</b></summary>
#
# **Last Execution:** 2026-08-04
#
# | Package | Version |
# |---------|---------|
# | **nnsight** | **0.8** |
# | Python | 3.12.13 |
# | torch | 2.13.0+cu126 |
# | transformer-lens | 3.0 |
#
# </details>

# %% [markdown]
# # Emergent World Models in Othello-GPT

# %% [markdown]
# ## Introduction

# %% [markdown]
# ⚫⚪ **Othello-GPT** is a transformer trained on one thing: sequences of legal Othello moves.
# It has never seen a board. It was not told the rules, the objective, or that the numbers it is
# predicting refer to squares in a grid. It is next-token prediction on a list of integers.
#
# It nevertheless plays legal Othello — and the reason is that it built a **model of the board**
# internally. Li et al. showed this by training probes on its activations that recover the state
# of all 60 squares, and then *editing* the probed representation and watching the model's
# predictions change to match the edited board. The world model is not decoration; it is what the
# predictions are computed from.
#
# Their probes had to be nonlinear, which was a puzzle — a linear representation is the usual
# expectation. Neel Nanda's follow-up resolved it: the world model *is* linear, in the right
# coordinates. Not "black disc / white disc / blank", but **"mine / theirs / blank"**, relative to
# whoever is about to move. Othello is symmetric between the players, and the model plays both, so
# the useful feature is the one that does not care which colour you are.
#
# 📗 [*Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task*](https://arxiv.org/abs/2210.13382)
# (Li, Hopkins, Bau, Viégas, Pfister, Wattenberg, ICLR 2023), and Nanda, Lee, Wattenberg,
# [*Emergent Linear Representations in World Models of Self-Supervised Sequence Models*](https://arxiv.org/abs/2309.00941)
# (BlackboxNLP 2023).
#
# We will implement Othello from scratch, confirm the model plays legally, train both probes to
# see the mine/theirs gap directly, and then intervene along the probe's directions to show the
# representation is causal.

# %% [markdown]
# ## Setup: wrapping a TransformerLens model

# %% [markdown]
# One thing to explain before any code. [**TransformerLens**](https://github.com/TransformerLensOrg/TransformerLens)
# is an interpretability library that reimplements transformers in its own format, with named
# `HookPoint` modules at every interesting intermediate value. A lot of published
# interpretability work ships weights in that format, and Othello-GPT is one of them — there is no
# HuggingFace release of this model, only `NeelNanda/Othello-GPT-Transformer-Lens`.
#
# That is not a problem, because `nnsight.NNsight` traces **any** `torch.nn.Module`. So we load the
# model with TransformerLens — purely as a weight loader and architecture — and inspect it with
# `nnsight`. TransformerLens's hook points are ordinary modules, so they show up as envoy paths
# and their `.output` is the value the hook would have received:
#
# | TransformerLens hook | nnsight path |
# |---|---|
# | `blocks.6.hook_resid_post` | `model.blocks[6].hook_resid_post.output` |
# | `blocks.3.attn.hook_z` | `model.blocks[3].attn.hook_z.output` |
# | `blocks.3.attn.hook_pattern` | `model.blocks[3].attn.hook_pattern.output` |
#
# The same applies to any model you have lying around — your own architecture, a research
# codebase, a checkpoint from a paper. `NNsight` wraps the module; nothing has to be ported.

# %% [markdown]
# If using Colab, install NNsight and TransformerLens:
# ```
# !pip install -U nnsight transformer_lens
# ```

# %%
try:
    import google.colab
    is_colab = True
except ImportError:
    is_colab = False

if is_colab:
    !pip install -U nnsight transformer_lens

# %%
import random

import torch
import torch.nn.functional as F
import nnsight
from transformer_lens import HookedTransformer

hooked = HookedTransformer.from_pretrained("othello-gpt", device="cuda").eval()
model = nnsight.NNsight(hooked)

N_CTX = hooked.cfg.n_ctx
print(f"{hooked.cfg.n_layers} layers, d_model={hooked.cfg.d_model}, "
      f"vocab={hooked.cfg.d_vocab}, context={N_CTX} moves")

# %% [markdown]
# 61 tokens for a game with 64 squares: the four centre squares start occupied and are never
# played, leaving 60 possible moves plus one token for "pass".

# %% [markdown]
# ## Othello, from scratch

# %% [markdown]
# We need the rules ourselves — to generate games, and more importantly to know the true board
# state at every step so we have something to probe *for*.
#
# A move is legal if it flanks at least one unbroken line of opponent discs between the new disc
# and one of your own. All flanked discs flip. Black moves first; a player with no legal move
# passes.

# %%
CENTRE = {(3, 3), (3, 4), (4, 3), (4, 4)}
SQUARES = [(r, c) for r in range(8) for c in range(8) if (r, c) not in CENTRE]
SQUARE_TO_TOKEN = {square: i + 1 for i, square in enumerate(SQUARES)}   # token 0 is "pass"
TOKEN_TO_SQUARE = {token: square for square, token in SQUARE_TO_TOKEN.items()}
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

BLACK, WHITE = 1, -1


def new_board():
    board = [[0] * 8 for _ in range(8)]
    board[3][3] = board[4][4] = WHITE
    board[3][4] = board[4][3] = BLACK
    return board


def flipped_by(board, row, col, player):
    """Discs that would flip if `player` played (row, col) — empty if the move is illegal."""
    if board[row][col] != 0:
        return []
    captured = []
    for d_row, d_col in DIRECTIONS:
        line, r, c = [], row + d_row, col + d_col
        while 0 <= r < 8 and 0 <= c < 8 and board[r][c] == -player:
            line.append((r, c))
            r, c = r + d_row, c + d_col
        if line and 0 <= r < 8 and 0 <= c < 8 and board[r][c] == player:
            captured += line
    return captured


def legal_moves(board, player):
    return [sq for sq in SQUARES if flipped_by(board, *sq, player)]


def apply_move(board, row, col, player):
    for r, c in flipped_by(board, row, col, player):
        board[r][c] = player
    board[row][col] = player


def random_game(rng):
    """Play random legal moves. Returns move tokens, the board after each, and who moved."""
    board, player = new_board(), BLACK
    moves, boards, movers = [], [], []
    while True:
        legal = legal_moves(board, player)
        if not legal:
            player = -player
            if not legal_moves(board, player):
                break
            continue
        row, col = rng.choice(legal)
        movers.append(player)
        apply_move(board, row, col, player)
        moves.append(SQUARE_TO_TOKEN[(row, col)])
        boards.append([row[:] for row in board])
        player = -player
    return moves, boards, movers


def show(board, marks=()):
    """Print a board. `marks` are squares to highlight (e.g. predicted moves)."""
    marks = set(marks)
    glyph = {BLACK: " ●", WHITE: " ○", 0: " ·"}
    print("   " + "".join(f" {c}" for c in "abcdefgh"))
    for r in range(8):
        cells = ["*" + glyph[board[r][c]][1] if (r, c) in marks else glyph[board[r][c]]
                 for c in range(8)]
        print(f" {r + 1} " + "".join(cells))


moves, boards, movers = random_game(random.Random(0))
print(f"a random game: {len(moves)} moves, first tokens {moves[:6]}")
show(boards[10])

# %% [markdown]
# ## The model plays legal Othello

# %% [markdown]
# The premise of the whole line of work. At every position we know the true board, so we know the
# legal moves; the question is whether the model's top prediction is one of them.

# %%
def sample_games(n, seed=0, min_length=N_CTX + 1):
    rng = random.Random(seed)
    games = []
    while len(games) < n:
        moves, boards, movers = random_game(rng)
        if len(moves) >= min_length:
            games.append((moves[:N_CTX], boards[:N_CTX], movers[:N_CTX + 1]))
    return games


games = sample_games(64, seed=0)
tokens = torch.tensor([g[0] for g in games], device="cuda")

with torch.no_grad(), model.trace(tokens):
    logits = model.output.save()

legal_hits = total = 0
for index, (moves, boards, movers) in enumerate(games):
    for t in range(N_CTX - 1):
        legal = {SQUARE_TO_TOKEN[sq] for sq in legal_moves(boards[t], movers[t + 1])}
        legal_hits += logits[index, t].argmax().item() in legal
        total += 1

print(f"top-1 prediction is a legal move: {legal_hits / total:.4f}  ({total} positions)")

# %% [markdown]
# Essentially always. Nothing in the training data said what "legal" means — the model was fed
# integer sequences — so whatever it is computing has the structure of the game in it.

# %% [markdown]
# ## Probing for the board

# %% [markdown]
# Now the question Li et al. asked: is the board *represented*, or is legality being computed some
# other way? A probe answers this by trying to read the board out of the residual stream. If a
# simple classifier on the activations recovers all 60 squares, the information is there and it is
# there in an accessible form.
#
# We collect the residual stream halfway up the model — layer 6 of 8 — for a few hundred games.
# One trace per batch, one saved tensor.

# %%
LAYER = 6

games = sample_games(500, seed=1)
tokens = torch.tensor([g[0] for g in games], device="cuda")

activations = []
for start in range(0, len(games), 100):
    with torch.no_grad(), model.trace(tokens[start:start + 100]):
        activations.append(model.blocks[LAYER].hook_resid_post.output.save())
activations = torch.cat(list(activations)).float()

print("activations:", tuple(activations.shape))

# %% [markdown]
# Two sets of labels for the same activations, and the whole argument is in the difference
# between them. Both describe the board after move $t$, per square, as one of three classes:
#
# - **black / white / blank** — the absolute state, the obvious choice;
# - **mine / theirs / blank** — the same board relabelled relative to whoever moves next.

# %%
mine_labels = torch.zeros(len(games), N_CTX, 60, dtype=torch.long)
colour_labels = torch.zeros(len(games), N_CTX, 60, dtype=torch.long)

for index, (moves, boards, movers) in enumerate(games):
    for t in range(N_CTX):
        to_move = movers[t + 1]
        for square_index, (row, col) in enumerate(SQUARES):
            disc = boards[t][row][col]
            mine_labels[index, t, square_index] = 0 if disc == 0 else (1 if disc == to_move else 2)
            colour_labels[index, t, square_index] = 0 if disc == 0 else (1 if disc == BLACK else 2)

mine_labels, colour_labels = mine_labels.cuda(), colour_labels.cuda()

# %% [markdown]
# The probe is deliberately the weakest thing that could work: one linear map from the 512-dim
# residual stream to 3 logits, per square, no hidden layer and no nonlinearity. If it succeeds,
# the board is linearly encoded.

# %%
N_TRAIN = 400

def train_probe(labels, name, steps=600):
    train_x = activations[:N_TRAIN].reshape(-1, 512)
    train_y = labels[:N_TRAIN].reshape(-1, 60)
    test_x = activations[N_TRAIN:].reshape(-1, 512)
    test_y = labels[N_TRAIN:].reshape(-1, 60)

    probe = torch.zeros(512, 60, 3, device="cuda")
    torch.nn.init.normal_(probe, std=0.02)
    probe.requires_grad_(True)
    optimizer = torch.optim.AdamW([probe], lr=1e-2, weight_decay=0.01)

    for _ in range(steps):
        predictions = torch.einsum("bd,dsc->bsc", train_x, probe)
        loss = F.cross_entropy(predictions.reshape(-1, 3), train_y.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        accuracy = (torch.einsum("bd,dsc->bsc", test_x, probe).argmax(-1) == test_y).float().mean()
    print(f"{name:>22}: {accuracy:.4f}")
    return probe.detach()


print("held-out probe accuracy over all 60 squares")
mine_probe = train_probe(mine_labels, "mine / theirs / blank")
colour_probe = train_probe(colour_labels, "black / white / blank")

# %% [markdown]
# There it is. The same activations, the same probe architecture, the same amount of training —
# and relabelling the board relative to the player to move takes a linear probe from mediocre to
# near-perfect.
#
# This is the correction Nanda's paper makes to Li et al. The original conclusion was that the
# world model is nonlinear, because a linear probe on *colour* does poorly. But the model plays
# both sides, and a feature meaning "this square is occupied by my opponent" is useful on every
# move regardless of colour, whereas "this square is black" is useful only half the time. The
# representation was linear the whole time; the coordinates were wrong.

# %% [markdown]
# ## The board is causal, not decorative

# %% [markdown]
# A probe reading out the board only shows the information is present. It could still be a
# by-product — computed and ignored. The test is to **edit** it: change the represented board, and
# see whether the model's predictions follow the board we wrote rather than the game it was given.
#
# The edit direction comes straight from the probe. To turn square $s$ from *theirs* into *mine*,
# push the residual stream along $w_{s,\text{mine}} - w_{s,\text{theirs}}$ — the direction that
# moves that square's classification from one class to the other, and nothing else.

# %%
edit_direction = (mine_probe[:, :, 1] - mine_probe[:, :, 2]).T       # [60, 512]
edit_direction = edit_direction / edit_direction.norm(dim=-1, keepdim=True)

# %% [markdown]
# To measure the effect we need positions where flipping one disc genuinely changes the game.
# For each case we pick a square held by the opponent, flip it to the current player, and keep the
# case only if this both **creates** legal moves and **destroys** others. Those two sets of moves
# are the measurement: if the edit works, the model should start predicting the created ones and
# stop predicting the destroyed ones.

# %%
def build_cases(n, seed=1234):
    rng = random.Random(seed)
    cases = []
    while len(cases) < n:
        moves, boards, movers = random_game(rng)
        if len(moves) < N_CTX + 1:
            continue
        t = rng.randrange(20, N_CTX)
        board, to_move = boards[t], movers[t + 1]
        for square_index, (row, col) in enumerate(SQUARES):
            if board[row][col] != -to_move:                 # want one of *their* discs
                continue
            edited = [r[:] for r in board]
            edited[row][col] = to_move
            after = {SQUARE_TO_TOKEN[sq] for sq in legal_moves(edited, to_move)}
            before = {SQUARE_TO_TOKEN[sq] for sq in legal_moves(board, to_move)}
            if len(after - before) >= 2 and len(before - after) >= 1:
                cases.append(dict(moves=moves[:N_CTX], t=t, square=square_index,
                                  board=board, edited=edited, to_move=to_move,
                                  created=sorted(after - before), destroyed=sorted(before - after)))
                break
    return cases


cases = build_cases(128)
case_tokens = torch.tensor([c["moves"] for c in cases], device="cuda")
rows = torch.arange(len(cases), device="cuda")
positions = torch.tensor([c["t"] for c in cases], device="cuda")
squares = torch.tensor([c["square"] for c in cases], device="cuda")

def probability_mass(logits):
    """Average probability on moves the edit creates, and on moves it destroys."""
    probs = logits[rows, positions].softmax(-1)
    created = torch.stack([probs[i, torch.tensor(c["created"], device="cuda")].sum()
                           for i, c in enumerate(cases)])
    destroyed = torch.stack([probs[i, torch.tensor(c["destroyed"], device="cuda")].sum()
                             for i, c in enumerate(cases)])
    return created.mean().item(), destroyed.mean().item()

# %% [markdown]
# The intervention itself: add the edit direction at the probed position, scaled relative to the
# residual stream's own norm so the perturbation is comparable across positions. We apply it
# across layers 4–6, since the board representation is distributed over several layers rather than
# written once.

# %%
def run(alpha, layers=(4, 5, 6)):
    with torch.no_grad(), model.trace(case_tokens):
        for layer in layers:
            resid = model.blocks[layer].hook_resid_post.output
            scale = resid[rows, positions].norm(dim=-1, keepdim=True)
            model.blocks[layer].hook_resid_post.output[rows, positions] = (
                resid[rows, positions] + alpha * scale * edit_direction[squares]
            )
        logits = model.output.save()
    return logits


print(f"{'edit strength':>14} | {'P(created moves)':>17} | {'P(destroyed moves)':>19}")
print("-" * 58)
baseline_logits = run(0.0)
created, destroyed = probability_mass(baseline_logits)
print(f"{'none':>14} | {created:>17.4f} | {destroyed:>19.4f}")
for alpha in [0.125, 0.25, 0.5, 1.0]:
    created, destroyed = probability_mass(run(alpha))
    print(f"{alpha:>14.3f} | {created:>17.4f} | {destroyed:>19.4f}")

# %% [markdown]
# Both quantities move by more than two orders of magnitude, in opposite directions and at the
# same time. Moves that are legal only on the *edited* board go from ~0.05% of the model's
# probability mass to ~20%; moves that were legal only on the *real* board collapse from ~10% to
# ~0.04%.
#
# The model is not confused by the intervention — it is playing a different, coherent game, the
# one implied by the board we wrote into it. That is what makes this a world model rather than a
# correlate.

# %% [markdown]
# ### One position, in full
#
# The aggregate is the evidence; a single case is what makes it legible.

# %%
index = max(range(len(cases)), key=lambda i: len(cases[i]["created"]))
case = cases[index]
edited_logits = run(0.5)

print(f"the model has seen {case['t'] + 1} moves; "
      f"{'black' if case['to_move'] == BLACK else 'white'} to play")
print("\nreal board (* = the disc we will flip)")
show(case["board"], marks=[SQUARES[case["square"]]])
print("\nedited board — that disc now belongs to the player to move")
show(case["edited"], marks=[SQUARES[case["square"]]])

legal_real = {SQUARE_TO_TOKEN[sq] for sq in legal_moves(case["board"], case["to_move"])}
legal_edited = {SQUARE_TO_TOKEN[sq] for sq in legal_moves(case["edited"], case["to_move"])}
name = lambda token: f"{chr(97 + TOKEN_TO_SQUARE[token][1])}{TOKEN_TO_SQUARE[token][0] + 1}"

print(f"\nthe edit creates {[name(t) for t in case['created']]} "
      f"and destroys {[name(t) for t in case['destroyed']]}")

for label, source in [("before the edit", baseline_logits), ("after the edit", edited_logits)]:
    top = source[index, case["t"]].topk(5).indices.tolist()
    print(f"\ntop-5 predicted moves {label}: {[name(t) for t in top]}")
    print(f"  legal on the REAL board   : {sum(t in legal_real for t in top)}/5")
    print(f"  legal on the EDITED board : {sum(t in legal_edited for t in top)}/5")
    print(f"  moves the edit created    : {sum(t in case['created'] for t in top)}/5")

# %% [markdown]
# Before the edit the model plays the real game. After it, its top five are moves that are legal
# on the board we wrote — including ones that are *only* legal there, which it had no reason to
# consider a moment earlier.

# %% [markdown]
# ## Conclusion

# %% [markdown]
# 🎉 A model trained on integer sequences, with no notion of a board, turns out to keep one — in a
# linear code, relative to whose turn it is — and to compute its predictions from it. Overwrite
# the board and the predictions change to match.
#
# On the `nnsight` side, the thing to take from this notebook is that none of it required the
# model to be in a supported format. Othello-GPT exists only as TransformerLens weights;
# `nnsight.NNsight` wrapped the module and every hook point became an addressable path, for
# reading and for writing. The same holds for a model you are training yourself — see
# [Progress Measures for Grokking](grokking-progress-measures.ipynb), which does exactly that with a plain
# PyTorch module.
#
# Related: [Probing](../tutorials/probing/logit_lens.ipynb) for reading representations, and
# [Model Editing](../../features/7_model_editing.ipynb) for making interventions permanent.

# %% [markdown]
# ## References
#
# - Li, Hopkins, Bau, Viégas, Pfister, Wattenberg, [*Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task*](https://arxiv.org/abs/2210.13382), ICLR 2023
# - Nanda, Lee, Wattenberg, [*Emergent Linear Representations in World Models of Self-Supervised Sequence Models*](https://arxiv.org/abs/2309.00941), BlackboxNLP 2023
# - Model weights: [`NeelNanda/Othello-GPT-Transformer-Lens`](https://huggingface.co/NeelNanda/Othello-GPT-Transformer-Lens)
# - Adapted from the TransformerLens [Othello-GPT demo](https://github.com/TransformerLensOrg/TransformerLens/blob/main/demos/Othello_GPT.ipynb)

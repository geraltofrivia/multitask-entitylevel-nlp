# TODOs

## 08-07-2022

- [x] Run baseline with more CRAC things
- [x] Why is TRIM not working
- [x] Effects of scheduling. Try with different schedules
- [x] Run a grid of scheduling experiments - wire it up
- [x]

## 18-07-2022

- [x] Talking about **speaker_ids**

- Embeddings be domain specific
- So its possible that in some domain, we have contatenated this, and not in some other domains
- Which creates a mismatch in slow_antecedent_scorer (Ctrl+f this in model).
- So we need to create faux speaker IDs for EVERYTHING. And a flag to either include them or not at all
    - an approach for that would be to add a n+1 embedding and use the other embedding
      in cases where domain 2 has no speakers
    - alternatively just put [0,0,0...] everywhere.
    - THIS NEEDS TO BE DONE TO MAKE THE MODEL WORK WITH SPEAKERS **ded**

## 20-07-2022

So that seems to be done. Now there are two bugs that we need to fix. Urgently.

1. There's something very wrong with persuasion.
    1. First step is to try re-running the baseline on the current commit without speakers using --ignore-speakers flag
    2. Re-running the baseline associated with the right commit and seeing if that does something. You better hope it does
    3. Then its a matter of opening diffs and seeing what went wrong. Or preproc to see what went wrong.
       Basically I have no idea.
2. Ontonotes full does not even fit in memory. Not even the train set.
   We really do need to move to the HOI way of doing things.
   Shall we just do that hereon?

## 22-07-2022

So after three days, the bug is fixed. And what a bug fix wow fuck me sideways. Anyway.
We'll be running a few more experiments. All on persuasion (or light) and trying to maximize the numbers we're getting
Specifically

1. cr-pr-ccper-cr-pr-on-lrc-lspru
2. full onto alongwith persuasion (on the 32G GPU I just spotted)
3. switchboard baselines
4. throw in an NER dataset.
5. NER onto vs Coref onto

Other TODOs

- [x] Do loss scales (defined in the task triple) change anything?
- [x] coref num speakers variable... is it needed? it shouldn't be.

## 01-08-2022

- [x] Implement domain adversarial learning thing
- [x] Change encoder to something autoregressive
- [x] Analyse predictions of model - look at what samples are predicted. Can we curtail spans by POS tags?
- [x] Feed spans from the data iter part instead of within the model
- [x] Integrate Univ. Anaphora Scorer

## 03-08-2022

- [x] Make different chunks of the MTL model
- [x] Encoder chunk (that extends Tranformers sure)
- [x] NER chunk
- [x] Coref Chunk
- [x] Integration Logic remains in MTL but modules do their job well.

--------    

- [x] Compute Baseline Results
- [x] Implement Shared Dense Layers (2 layer dense net with same or half dimensionality)

## 16-08-2022

- [x] Figure out the deal with CoNLL12
- [ ] Implement Shared Pruner (integrated in there)

### Refactoring for multiple NER tags per span

- [x] Figure out how loss scales work, and simplify it\
- [ ] Recheck if tokenizer is working as intended (do we need special chars or not)

## 21-08-2022

- [x] Fix MEMORY LEAK
- [x] Add a POS tagging task (which is token level, for a change. yay ^^)
- [x] Can we make the sample stuff resample every epoch?
- [x] Figure out how sampling ratios work
- [x] Simulate the loss thing (all entries get a 1)
- [x] dwie baselines
- [x] Check if ON works better with SpanBERT base or large

## 23-08-2022

- [ ] figure out whats going wrong with NER eval (acc stuff)

-----

# Leftovers

- [ ] Implement Shared Pruner (integrated in there)
- [ ] Recheck if tokenizer is working as intended (do we need special chars or not)
- [ ] Make setup.sh conditioned and verbose
- [x] Should we make a new repo for preproc data management?: NO!

# General

- [ ] Find a way to easily run sweeps

# ACE

- [ ] Figure out NER metrics and clamp them down tight.
- [ ] Get ACE Sweep Started
- [ ] Find a place to note down important experiment results

# Ontonotes

So I'm still short of the baseline. Time to fork out and try to get baseline Ontonotes working well.
First step: get the current best result out. Sweep.

- [ ] Run a bert-base-uncased sweep
- [ ] Mimic parameter init (and see if performance rises or falls)
- [ ] Okay, its time to write a OAR management class/thing

----

# 21-09-2022

- [!] **Redo all NER runs** (We are working with shared pruner!)
- [ ] Add a flag to disassociate NER with pruner stuff.
# TODOs

## 08-07-2022

[ ] Run baseline with more CRAC things
[ ] Why is TRIM not working
[ ] Effects of scheduling. Try with different schedules
[ ] Run a grid of scheduling experiments - wire it up
[ ]

## 18-07-2022

[ ] Talking about **speaker_ids**

- Embeddings be domain specific
- So its possible that in some domain, we have contatenated this, and not in some other domains
- Which creates a mismatch in slow_antecedent_scorer (Ctrl+f this in model).
- So we need to create faux speaker IDs for EVERYTHING. And a flag to either include them or not at all
    - an approach for that would be to add a n+1 embedding and use the other embedding
      in cases where domain 2 has no speakers
    - alternatively just put [0,0,0...] everywhere.
    - THIS NEEDS TO BE DONE TO MAKE THE MODEL WORK WITH SPEAKERS **ded**

# 20-07-2022

So that seems to be done. Now there are two bugs that we need to fix. Urgently.

1. There's something very wrong with persuasion.
    1. First step is to try re-running the baseline on the current commit without speakers using --ignore-speakers flag
    2. Re-running the baseline associated with the right commit and seeing if that does something. You better hope it does
    3. Then its a matter of opening diffs and seeing what went wrong. Or preproc to see what went wrong.
       Basically I have no idea.
2. Ontonotes full does not even fit in memory. Not even the train set.
   We really do need to move to the HOI way of doing things.
   Shall we just do that hereon?
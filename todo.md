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
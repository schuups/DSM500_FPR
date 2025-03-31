# Final Project Report TODO list

## 00 head ✅
- [x] 00 - Cover page ✅
- [x] 01 - Abstract ✅
- [ ] 02 - Thanks and acknowelgments 
- [x] 03 - Word count (final update) ✅

## Table of contents 🟠
- [ ] Looks good, e.g. word upper-casing (final check) 📌 

## 01 intro 🟠 (might benefit from adding links and citations)
- [x] 00 - Introduction text ✅
- [x] 01 - Project aims ✅
- [x] 02 - Research question ✅
- [x] 03 - Professional context ✅
    - [x] Infrastructure constraints ✅
- [x] 04 - Statement of relevance ✅
- [x] 05 - Ethical and legal considerations ✅
- [x] 06 - Statement on the use of LLM-based services 🟠 (add a mention for code sketching and alternative to search enginers .. example cartography)
- [ ] [07 - Project design and implementation 🪚❓]

## 02 background ✅
- [x] 00 - Data-driven weather forecasting ✅
- [x] 01 - Models for data-driven weather forecasting ✅
- [x] 02 - Graph Neural Networks (GNNs) ✅
    - [x] Review of relevant publications ✅
- [x] 03 - Datasets ✅
    - [x] Bias (add a mention about the soutern hemisphere and extreme events being less represented, and that this is a factor relevant to bias - ) ✅
    - [x] Add small ref to ERA5 being the "GOLD STANDARD" in the field ✅
- [x] 04 - Common variables of interest ✅
- [x] 05 - Common evaluation methods (and metrics?) ✅
- [x] 06 - Common methods for models comparison ✅

## 03 methods 🔴
- [x] 00 - Dataset ✅
    - [x] Dataset splits ✅
    - [x] Climatology ✅
- [ ] Codebase and model architecture
    - [x] code organisation ✅
    - [x] training setup ✅
    - [ ] model architecture 🔴
    - [x] increasing steps, 8 steps testing phase called validation ✅
    - [x] learning rate schedule ✅
- [ ] Results evaluation protocol 🟠 (some todos remaining!)
    - [x] Baselines (e.g. FCN) ✅
    - [x] Training goal: maximize training run, while balancing between under-fitting and over-fitting, and related ✅
    - [x] Run multiple training with different seeds, so to allow models to see different sequences of data ✅
    - [x] results interpretation: statistical methodology ✅
    - [x] results interpretation: visuals (scorecards + grids + sample plots in appendix + video) ✅
- [x] Experimental approach ✅
- [x] Infra ✅
    - [x] SLURM: queueing system intro ✅
    - [x] wandb ✅
    - [x] Setup (HPC, SLURM, containers, git, wandb) ✅
- [ ] The methods need to discuss how to extrapolate the results obtained on a 30 minutes run, to suggest if GraphCast is a good option on longer trainings 🔴 

## 06 results ✅
- [x] 01 - GraphCast Dry-Run ✅
- [x] 02 - Inference framework  ✅
- [x] 03 - Refactoring effort ✅
- [x] 04 - GraphCast performance baseline recording ✅
- [x] 05 - Optimal learning schedule and commoong iteration count est. ✅
- [x] 06 - GraphCast enhancements ✅
- [x] 07 - Ablation Study --> Improved GraphCast performance recording ✅
- [x] 08 - FourCastNet ✅
- [x] 09 - Further Experimentation ✅
- [x] 10 - Improved performances ✅
- [x] 11 - Visualization products ✅
- [x] 12 - Quantitative results and statistical evaluation ✅

## 07 discussion ✅
- [x] 00 - Discussion text ✅

## 08 conclusion ✅
- [x] 00 - Conclusion text ✅


## References 🔴
- [ ] Check citation format is as required (final check) 📌

## 99 - appendices 🔴
- [ ] 00 - Abbreviations and definitions (final check) 📌
    - debug queues
    - rank and distributed primer, world_size

    - variables vs channels
    - batch size, global batch size, local batch size

    - TOP500

- [ ] 01 - A primer on GNNs
- [ ] 02 - Other weather datasets

---

## _later 🔴
- [ ] Check that all content has been integrated or discarder
- [ ] Check headers capitalization
- [ ] Verify that all acronyms are provided only once.
- [ ] Remove all TODOs
- [ ] Mention that to test the reproducibility, a new repo has been created from scratch and all the changes reimplemented. For the notebooks, most of the time they just have been copied as is, and thus paths in them might need to be updated if the reader wants to test them instead of just running them.
- [ ] Open access to github repo and wandb
- [ ] Add "for the reminder of this project, GraphCast refers to the NVIDIA GraphCast re-implementation in PyTorch"... clarification needed to facilitate the reader
---



# Video TODO list
- Show old gitlab (all the nodebooks), the recent github and the current github ... make ref to the current github being a refactoring of what done before done with the purposes of wrapping up many directions (brining focus back to the FPR) and confirm reproducibility.
- Show video of the rotating worlds
- Show slurm queue + explain why they are called clariden and santis
    - Explain HPL and GB runs and the fact that the system was mostly unavailable for the last weeks (show email?)
- Review research questions
    - The actual model improvement was therefore squeezered - propose that the title is updated: although the improvements I initiall envisioned were more on the model architecture, the improvements made still improved this GraphCast implementation 
- Show the different git repos (gitlab, github1, github2) ... for the purpose of reproducibility
- Although findings were less substantial than I initially expected, this project execution (which saw my kids neglected for the past many weekends) still brought new knowledge both for DS but also in the context of distributed computing, tremendous experience (e.g. time consuming errors, won't be repeated)

- Highlight that the code base was completely new to me at the beginning of the project.
- Review all results, one by one.

- The recent maintenance window saw me working on the last night until 5 am on a sunday night. I had to subsequenyl skip work on the next day.

- Taking an enterprise level codebase, review it in depth (often extracting parts of it to study the implemention and functionality) and enhance it - it is something I have never done before and that this MSc project pushed me to do. I am satisfied by having demostrated this ability. After the marking of this work I will be able to bring it to the next steps, in the context of my team at work, and look at broader and deeper enhancements such as the integration of other models (ref. papers).

- Ready to bring this into the team once it is marked, to continue the work with domain specialists.
- Review access invites to the github repos (current and past)

- The fact that I rerun everything in the past weeks makes me confident that the code runs and produces consistent results (w.r.t. reproducibility).

- Discuss how the achieved testing MSE has been improved over multiple iterations.
- Discuss how a substantial part of the project time has been spent on infrastructure related issues, not in the sense of system engineering issues, but to cope with unexpected variability. Show plot of the memory error corrections (image sent in Slack) and show the backup system prepared on GCP.

- Show all the checkpoints in vscode
- The reader is free to explore data (experiments results) in Wandb

- Describe how the literature employed (e.g. from arvix) is generally vetted, if not peer reviewed. Most of them go through the Supercomputing conference.

- Interestingly, it was not clear why the metric value increased considerably when the number of GPUs involved in the training increased. Mention how initially the tool did not average the testing MSE, but only reported the metric obtained on rank 0. Initially fixed I had to rollback to save time :) 
- Describe how the anylsis of the code was not given much space in the report as it was deemed not directly relevant to answer the quesitons. Two examples (show th code?):
    - max_edge_length function
    - add_edge_features

- Future works
    - Test CUDA Graphs?


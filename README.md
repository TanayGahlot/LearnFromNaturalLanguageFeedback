# LearnFromNaturalLanguageFeedback
Adaptation of large language models to downstream task has shown promising
results but it remains costly and resource intensive for majority of production
applications. In this work, we focus on improving the convergence rate to reduce
the cost of RL based fine-tuning method. Currently, majority of RL based finetuning
techniques focus on numerical rewards and ignore the natural language
feedback. We propose two techniques to combine natural language feedback with
numerical feedback in existing policy gradient setup. Firstly, we combine the
objective of policy gradient with imitation learning objective to get a speedup
of 1.5x. Secondly, we introduce expert trajectories into the episode to get 2x
improvement in convergence rate.

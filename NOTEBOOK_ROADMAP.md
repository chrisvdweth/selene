# Notebook Roadmap

This roadmap fills the main historical and conceptual gaps between classical statistical learning, early neural networks, deep learning, reinforcement learning, Transformers, and multimodal models.

## Coverage Levels

- **Dedicated:** the repository has a notebook centered on the topic.
- **Partial:** the topic is explained within another notebook but is not treated independently.
- **Missing:** no meaningful instructional coverage was found.

## Phase 1: Statistical Foundations

### 1. Bayes' Rule

- **Status:** Dedicated
- **Notebook:** `11-bayes_rule_basics.ipynb`
- **Scope:** conditional probability, priors, likelihood, evidence, posterior probability, worked diagnostic-test example, and a small simulation.
- **Existing connection:** `multinomial_naive_bayes_basics.ipynb`

### 2. Maximum Likelihood Estimation

- **Status:** Dedicated
- **Notebook:** `12-maximum_likelihood_estimation_basics.ipynb`
- **Scope:** likelihood versus probability, log-likelihood, Bernoulli and Gaussian derivations, numerical optimization, and the connection to cross-entropy.
- **Existing connections:** `multinomial_naive_bayes_basics.ipynb`, `ngram_language_models.ipynb`, and `time_series_analysis_overview.ipynb`

### 3. Expectation-Maximization

- **Status:** Dedicated
- **Notebook:** `14-expectation_maximization_basics.ipynb`
- **Scope:** latent variables, lower-bound intuition, E-step, M-step, convergence, and a Gaussian mixture model implemented from scratch.
- **Prerequisites:** Bayes' Rule and Maximum Likelihood Estimation

## Phase 2: Early Neural Networks

### 4. Perceptron

- **Status:** Dedicated
- **Notebook:** `35-perceptron_learning_algorithm.ipynb`
- **Scope:** binary linear classification, threshold activation, perceptron update rule, convergence intuition, and implementation from scratch.

### 5. XOR and the First Neural-Network Winter

- **Status:** Dedicated
- **Notebook:** `36-perceptron_xor_limitations.ipynb`
- **Scope:** linear separability, why a single-layer perceptron cannot represent XOR, a geometric demonstration, the role of Minsky and Papert, and how hidden layers solve XOR.
- **Prerequisite:** Perceptron

### 6. Neocognitron

- **Status:** Dedicated
- **Notebook:** `48-neocognitron_cnn_history.ipynb`
- **Scope:** simple and complex cells, hierarchical feature extraction, translation tolerance, relationship to Hubel-Wiesel findings, and influence on modern convolutional neural networks.

### 7. Backpropagation

- **Status:** Dedicated
- **Existing notebooks:** `backpropagation_basic_examples.ipynb`, `backpropagation_generalization.ipynb`, and `backpropagation_through_time_basics.ipynb`
- **Roadmap action:** add a short historical note covering Werbos, Rumelhart, Hinton, and Williams rather than creating another notebook.

## Phase 3: Sequence Learning and Kernel Methods

### 8. TD-Gammon

- **Status:** Dedicated
- **Notebook:** `52-td_gammon_temporal_difference_learning.ipynb`
- **Scope:** value functions, temporal-difference learning, self-play, TD(λ), neural value approximation, and a small tractable board-game demonstration rather than full backgammon.

### 9. Vanishing and Exploding Gradients

- **Status:** Dedicated
- **Notebook:** `51-vanishing_exploding_gradients.ipynb`
- **Scope:** repeated Jacobian products, activation saturation, numerical experiments across sequence lengths, gradient plots, clipping, initialization, normalization, residual connections, and gated recurrent units.
- **Existing connections:** `backpropagation_through_time_basics.ipynb`, `language_models_basics.ipynb`, and `transformers_basic_architecture.ipynb`

### 10. Long Short-Term Memory Networks

- **Status:** Dedicated
- **Notebook:** `57-lstm_from_scratch.ipynb`
- **Scope:** cell state, input/forget/output gates, forward equations, gradient-flow intuition, NumPy implementation, and comparison with a vanilla RNN.
- **Existing connections:** `text_classification_rnn.ipynb` and `word_text_embeddings_overview.ipynb`

### 11. Support Vector Machines

- **Status:** Dedicated
- **Notebook:** `20-support_vector_machines_basics.ipynb`
- **Scope:** maximum-margin classification, support vectors, hard and soft margins, hinge loss, kernels, and scikit-learn examples with decision boundaries.

## Phase 4: Deep Learning Revival and Vision

### 12. Restricted Boltzmann Machines

- **Status:** Dedicated
- **Notebook:** `49-restricted_boltzmann_machines.ipynb`
- **Scope:** energy-based models, visible and hidden units, Gibbs sampling, contrastive divergence, reconstruction, and historical links to deep belief networks.
- **Note:** present RBMs as one important part of the deep-learning revival, not as its sole starting point.

### 13. ImageNet and AlexNet

- **Status:** Dedicated
- **Notebook:** `50-imagenet_alexnet_deep_learning_breakthrough.ipynb`
- **Scope:** the ImageNet dataset and challenge, AlexNet architecture, ReLU, dropout, GPU training, data augmentation, and the 2012 result's historical impact.
- **Implementation:** train a compact AlexNet-inspired model on a manageable image dataset instead of full ImageNet.

## Phase 5: Deep Reinforcement Learning

### 14. Playing Atari with Deep Reinforcement Learning

- **Status:** Dedicated
- **Notebook:** `53-deep_q_learning_atari.ipynb`
- **Scope:** Q-learning, neural Q-functions, replay memory, target networks, frame preprocessing, reward clipping, and the original DQN contribution.
- **Implementation:** use a lightweight Gymnasium environment by default, with Atari as an optional extension.

### 15. AlphaGo

- **Status:** Dedicated
- **Notebook:** `54-alphago_policy_value_mcts.ipynb`
- **Scope:** policy networks, value networks, Monte Carlo tree search, self-play, supervised pretraining, reinforcement learning, and the relationship to AlphaGo Zero.
- **Implementation:** demonstrate MCTS and learned policies on a small board game.

## Phase 6: Transformers and Foundation Models

### 16. Attention Is All You Need

- **Status:** Dedicated across multiple notebooks
- **Existing notebooks:** `attention_mha_basics.ipynb`, `transformers_basic_architecture.ipynb`, `positional_encodings_original_transformer.ipynb`, and `machine_translation_transformers.ipynb`
- **Roadmap action:** add a short index notebook linking the components in paper order rather than duplicating their content.

### 17. GPT-3 and In-Context Learning

- **Status:** Dedicated
- **Notebook:** `88-gpt3_in_context_learning.ipynb`
- **Scope:** decoder-only Transformers, scaling, autoregressive pretraining, zero-shot/one-shot/few-shot prompting, in-context learning, limitations, and the progression from GPT-1 and GPT-2.
- **Existing connections:** `transformers_basic_architecture.ipynb` and `llm_building_gptstyle_llm_from_scratch.ipynb`
- **Implementation:** use a small open model to illustrate prompting behavior; do not attempt GPT-3-scale training.

### 18. CLIP

- **Status:** Dedicated
- **Notebook:** `97-clip_contrastive_language_image_pretraining.ipynb`
- **Scope:** paired image-text data, dual encoders, cosine similarity, contrastive loss, zero-shot classification, retrieval, and limitations of web-scale supervision.
- **Implementation:** use a pretrained open CLIP model with a small image set.

## Recommended Build Order

1. `11-bayes_rule_basics.ipynb`
2. `12-maximum_likelihood_estimation_basics.ipynb`
3. `14-expectation_maximization_basics.ipynb`
4. `35-perceptron_learning_algorithm.ipynb`
5. `36-perceptron_xor_limitations.ipynb`
6. `20-support_vector_machines_basics.ipynb`
7. `51-vanishing_exploding_gradients.ipynb`
8. `57-lstm_from_scratch.ipynb`
9. `48-neocognitron_cnn_history.ipynb`
10. `49-restricted_boltzmann_machines.ipynb`
11. `50-imagenet_alexnet_deep_learning_breakthrough.ipynb`
12. `52-td_gammon_temporal_difference_learning.ipynb`
13. `53-deep_q_learning_atari.ipynb`
14. `54-alphago_policy_value_mcts.ipynb`
15. `88-gpt3_in_context_learning.ipynb`
16. `97-clip_contrastive_language_image_pretraining.ipynb`

This order prioritizes conceptual prerequisites and keeps expensive historical systems reproducible through smaller educational demonstrations.
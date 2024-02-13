This repository maintains brief summary of my readings. Most of them are papers, but also includes remarkable things other than paper.

## Papers

### State Space Model

#### LSSL
- Gu et al., "Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers
  - Introduced the Linear State-Space Layer (LSSL) that maps a sequence $u$ to $y$, where $u$ is a continous data. (e.g., audio data)
  - LSSL can be adapted to descrete data $u$ by chunking the data with size $\Delta t$. (e.g., text data)
  - LSSL computes $\dot x = Ax + Bu$ and $y = Cx + Du$, where $x$ represents the previous state, $\dot x$ represents the current state, and $A, B, C, D$ are learnable parameters. (Be careful that $x$ in $y$ is equal to $\dot x$. I have no idea why the authors use different notation.) It updates the current state which is a weighted sum of previous state and current input. It outputs the prediction which is a weighted sum of current state and current input. The term $Du$ is for residual connection, so it can be omitted for simplicity.
  - LSSL can be used just like RNN when inference, while it can be efficiently trained in parallel.
  - The parallelization is done by the convolution (sum of multiplications): $y = [CB, CAB, CA^2B, \dots, CA^{N-1}B] * u$. However, the naive computation of the convolution takes $O(N^2L)$ time. A theoretically efficient algorithm can compute it in quasi-linear time and space $\tilde O(N+L)$ (but not implemented).
  - In summary, LSSL is an interesting model that can be trained in parallel and that can conduct inference just like RNN. This resolves the inefficient training problem of RNN. At the same time, LSSL provides efficient inference in terms of both time and memory just like RNN.

#### S4
- Gu, Goel, and Re, "Efficiently Modeling Long Sequences with Structured State Spaces
  - Proposed effficient implementation of LSSL, where the convolution can be computed in $\tilde O(N+L)$ time using $O(N+L)$ space.
  - The idea is reducing the SSM to the computation of a Cauchy kernel by conditioning $A$ with a low-rank correction. (I did not look into the method.)
  - The resulting model architecture is called Structured State Space Sequence (S4) layer.
  - In short, S4 is a training-efficient version of LSSL.


#### H3
- Fu and Dao et al., "Hungry Hungry Hippos: Towards Language Modeling with State Space Models
  - This paper proposes a state space model (SSM) that can perform well on natural language understanding tasks.
  - The key idea is to add "copy" and "compare" functions to SSM. These two function is already provided by the attention operation, i.e., $QK^{-1}V$, where $QK^{-1}$ computes the similarity between $Q$ and $K$ (and thus comparing all pairwise tokens), and multiplying $V$ is a kind of copying.
  - The authors propose Hungry Hungry Hippos (H3) layer, which is two SSMs, one having shifted diagonal matrix $A$ (seems like it represents the previous state), another hvaing diagonal matrix $A$ (seems like it is used for pairwise comparison). Three projections $Q, K, V$ of input $x$ is fed into two SSMs.
  - There is a game called Hungry Hungry Hippos. In the game each person should collect marvels using a Hungry Hippo's mouse. (One marvel can go into one Hungry Hippo's mouse.) Maybe an SSM corresponds to a Hungry Hippo because each of $K$ and $SSM(K)V$ is fed to only one SSM.
  - The paper empirically shows that H3 outperforms S4, and that H3 performs as good as attention models.
  - This is the first work that well adapted SSM to natural language tasks.
 
#### Mamba
- Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces
  - This paper proposes a Selective State Space model, called Mamba, which has a structure combining H3 and gated MLP.
  - The idea is to add a model functionality of selective copying. This leads the model to work better for long seqeunces than other models inclusing H3 and Transformers.
  - Somehow, Mamba has lower perplexity than Transformers of the same size. At the same time, it has better latency and throughput. Mamba uses slightly more memory than Transformers with FlashAttention, where both FlashAttention and Mamba uses linear space.
  - Mamba will be the next Transformers.
 

### Multimodal (text, image) Model and Data

#### Conceptual Captions
- Sharma et al., "Conceptual Captions, A Cleaned, Hypernymed, Image Alt-text Dataset for Automatic Image Captionning", ACL 2018.
  - Presented a dataset of image caption annotations, called Conceptual Captions, which consists of 3.3M <image, description> pairs.
  - Used a Flume pipeline to extract, filter, and processes candidate <image, caption> pairs.
    - Image-based filtering: discards images based on encoding format, size, aspect ratio, and offensive content. Only keeps JPEG images where both dimensions are greater than 400 pixels, and the ratio of large to smaller dimension is no more than 2. Excludes images that trigger pornography or profanity detectors.
    - Text-based filtering: harvests Alt-text from HTML webpages. Analyzed candidate Alt-text using the Google Cloud Language APIs, including part-of-speech (POS), sentiment/polarity, and pornography/profanity annotations. Candidates with no determiner, no noun, or no preposition are discarded. Candidates with a nigh noun ratio are also discarded. Candidates with a high rate of token repetition are discarded. Candidates where the first word is not capitalized, or with too high capitalized-word ratio are discarded. Discarded candidates that contain tokens that are not appearing in the English Wikipedia at least 5 times. Candidates that score too high or too low on the polarity annotations, or trigger the pornography/profanity detectors, are discraded. Predefined boiller-plate prefix/suffix sequences matching the text are croppsed, e.g., "click to enlarge picture", "stock photo". Also drop text which begins/ends in certain patterns, e.g., "embedded image parmalink", "profile photo".
    - Image&Text-based filtering: filter out candidates for which none of the text tokens can be mapped to the content of the image.
    - Text transformation with hypernymization: noun modifiers of certain types (proper nouns, number, units) are removed; dates, durations, and preposition-based locations (e.g., "in Los Angeles") are removed; named-entities are identified, matched against the KG entries, and substitute with their hypernym; resulting coordination noun-phrases with the same head (e.g., "actor and actor") are resolved into a single-head, pluralized form (e.g., "actors"). Too short or inconsistent samples are discarded after transformation. Cluster all resolved entities (e.g., "actor", "dog", "neighborhood") and keep only candidates for which all detected types have a count of over 100.

#### ALIGN
- Jia et al., "Scaling Up Visual and Vision-Language Representation Learning with Noisy Text Supervision", ICML 2021.
  - Presented a nosiy dataset of 1.8B <image, alt-text> pairs, obtained without expensive filtering or post-processing steps in Conceptual Captions.
  - Apply simple frequency-based filtering.
    - Image-based filtering: remove pornographic images and keep only images whose shorter dimension is larger than 200 pixels and aspect ratio is smaller than 3. Images with more than 1000 associated alt-texts are discarded. Remove test images in ILSVRC-2012, Filckr30K, and MSCOCO.
    - Text-based filtering: exclude alt-texts that are shared by more than 10 images. Discard alt-texts that contain any rare token (outside of 100M most frequent unigrams and bigrams from the raw dataset), and those that are either too short (<3 unigrams) or too long (>20 unigrams).
  - Trained an embedding model called ALIGN (A Large-scale ImaGe and Noisy-text embedding).

#### CLIP
- Radfoard and Kim et at., "Learning Transferable Visual Models from Natural Language Supervision", PMLR 2021.
  - Created a new dataset of 400M <image, text> pairs, called WIT (WebImageText).
    - Search for pairs whose text includes one of a set of 500,000 queries. The base query list is all words occurring at least 100 times in the English version of Wikipedia. Then, the query is augmented with bi-grams with high pointwise mutual information as well as the names of all Wikipedia articles above a certain search volume. Finally all WordNet synsets not already in the query list are added.
    - Approximately class balance the results by including up to 20,000 pairs per query.
  - Trained a model called CLIP (Constrastive Language-Image Pre-training).

## Articles

## GiHub Repositories

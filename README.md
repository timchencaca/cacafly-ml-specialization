# cacafly-ml-specialization-demo3

This project demonstrates how to apply **supervised fine-tuning (SFT)** to a pretrained Gemini 2.0 Flash model using Vertex AI. The base model, originally trained for general-purpose language understanding, is fine-tuned on a custom binary classification task: determining whether two text segments are from the same movie review. The training dataset is constructed from the Cornell Movie Review corpus and includes carefully designed prompt-response pairs. Through fine-tuning, the model improves its accuracy from approximately 86% to over 93%, showcasing enhanced performance and task specialization on text similarity detection.

## Data Preprocessing (`demo3_data_prep.ipynb`)

This notebook prepares training data for fine-tuning the Gemini 2.0 Flash model. It processes raw movie reviews into prompt-response pairs formatted as JSONL files. As part of preprocessing, the notebook optionally uses Gemini itself to remove named entities (e.g., person names and movie titles) to reduce bias. The resulting data is suitable for supervised fine-tuning on binary classification tasks (e.g., identifying whether two sentences are from the same review).

### Requirements

Ensure the following dependencies are installed:

- `scikit-learn`
- `google-genai`
- `google-cloud-aiplatform`
- `google-cloud-discoveryengine`
- `vertexai`
- `json`
- `nltk`

### Steps Performed in the Script

#### 1. Environment Setup

- Install required libraries for Vertex AI and Gemini APIs.
- Configure GCP project and region.
- Initialize the Vertex AI SDK.


#### 2. Load Raw Movie Reviews
- **Source**: [Cornell Movie Review Polarity Dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/)
- Uses `sklearn.datasets.load_files` to load text files from the `txt_sentoken/` directory.
- Each review is converted into a JSONL prompt that instructs the model to remove named entities and movie titles.

#### 3. Generate Batch Cleaning Prompts
Each prompt follows this format:

> "Please remove all names of people and works of art (like movies, books, songs) from the following text. Keep grammar natural."

Example JSONL entry:

```json
{
  "id": 1,
  "request": {
    "contents": [
      {
        "parts": {
          "text": "Please remove... Original:\n<review_text>\nCleaned:"
        },
        "role": "user"
      }
    ]
  }
}
```
#### 4. Launch Gemini Batch Prediction Job
- Submits the prompt JSONL file to Gemini via Vertex AI using the `google-genai` client.
- The batch job processes the prompts to generate cleaned text outputs.
- Output files are stored in Cloud Storage at:

- `gs://cacafly-ml-specialization-dataset/movie_review/prediction-model-2025-06-18T03:43:44.162739Z/predictions.jsonl`


#### 5. Monitor Job Completion
- The script continuously polls the batch job status until it reaches a completed state (`SUCCEEDED`, `FAILED`, `CANCELLED`, or `PAUSED`):

```python
while job.state not in completed_states:
    time.sleep(30)
    job = client.batches.get(name=job.name)
```



## Model Objective (`demo3_gemini_finetune_similarity.ipynb`)

This notebook demonstrates how to fine-tune a Gemini model on structured prompt/response data for a binary semantic similarity task. The final model can be evaluated and deployed using Vertex AI Model Tuning APIs.

---

### Requirements

Ensure the following dependencies are installed:

- `scikit-learn`
- `google-genai`
- `google-cloud-aiplatform`
- `google-cloud-discoveryengine`
- `vertexai`
- `json`
- `nltk`
- `tqdm`
- `random`


---

### Step 1: Environment Setup

- Install required libraries for Vertex AI and Gemini APIs.
- Configure GCP project and region.
- Initialize the Vertex AI SDK.

---

### Step 2: Data Preparation

#### I. Load Cleaned Movie Reviews

- Loads **2,000 cleaned movie reviews** (1,000 originally positive, 1,000 originally negative), based on the [Cornell Movie Review dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/).
- These reviews were **preprocessed using Gemini 2.0 Flash to remove named entities and movie titles**.
- Cleaned data is retrieved from: `gs://cacafly-ml-specialization-dataset/movie_review/prediction-model-2025-06-18T03:43:44.162739Z/predictions.jsonl`


#### II. Construct Paired Samples for Fine-tuning

> ⚠️ **Important**:  
> The original dataset labels each full review as `positive` or `negative` **based on sentiment**.  
> However, the **fine-tuning task is a binary classification of paragraph *relatedness*** — whether two text segments come from the **same review**.

##### Sample Construction Logic

| Pair Type         | Count | Description                                      |
|-------------------|-------|--------------------------------------------------|
| Positive (`yes`)  | 1,000 | Two segments from the **same** movie review     |
| Negative (`no`)   | 1,000 | Two segments from **different** movie reviews   |
| **Total**         | 2,000 | Balanced for binary classification              |

- **Positive pairs**:
  - Select 500 reviews labeled as positive sentiment and 500 as negative sentiment.
  - For each review, randomly sample ~40% of sentences and reorder them in their original sequence.
  - Remove the last sentence of the first segment and the first sentence of the second segment to reduce direct continuity between segments.
  - Both segments come from the same review, labeled `yes`.

- **Negative pairs**:
  - Select another 500 positive sentiment and 500 negative sentiment reviews.
  - Similarly sample and reorder sentences.
  - Combine segments from different reviews, labeled `no`.

This approach balances lexical diversity and semantic clarity, preventing the model from relying on full-text duplication.


#### III. Train/Test Split

- The full dataset of 2,000 pairs is split into:
  - **Training set**: 70% (1,400 pairs)
  - **Test set**: 30% (600 pairs)

**Prompt Format**:

```
Paragraph A: [text A]
Paragraph B: [text B]

Are these paragraphs from the same movie review? Answer yes or no.
```

---

### Step 3: Pre-Finetune Model Evaluation 

- Use `gemini-2.0-flash` via Vertex AI for baseline inference.
- Load validation samples from a JSONL file, extracting prompts and ground-truth labels.
- Generate responses with `model.generate_content(prompt)`, where responses are interpreted as `yes` or `no`.
- Accuracy is computed over 600 manually verified validation samples.

---

### Step 4: Fine-tuning with Vertex AI

- Fine-tune the pre-trained `gemini-2.0-flash-001` model using the `vertexai.tuning.sft` API.
- Training dataset:  
  `gs://cacafly-ml-specialization-dataset/movie_review/train_rv_name_random_sent.jsonl`
- Fine-tuning parameters:
  - `epochs=5`
  - `adapter_size=4` (controls model adapter capacity)
  - `learning_rate_multiplier=3.0` (scales learning rate for tuning)
- Submit the training job with `sft.train()` which returns a tuning job handle.
- Poll the job status with `sft_tuning_job.refresh()` every 60 seconds until completion.
- Upon completion, retrieve:
  - Tuned model resource name (`sft_tuning_job.tuned_model_name`)
  - Model endpoint name for deployment (`sft_tuning_job.tuned_model_endpoint_name`)
  - Experiment metadata (`sft_tuning_job.experiment`)

This process adapts the base Gemini 2.0 flash model to the movie review relatedness classification task using your custom dataset.


---

### Step 5: Post-Finetune Model Evaluation

- Load the fine-tuned model endpoint using `vertexai.generative_models.GenerativeModel` with the endpoint name from the tuning job.
- Evaluation uses the same 600-sample validation set (`eval_rv_name_random_sent.jsonl`).
- For each sample:
  - Extract the prompt and expected answer (`yes` or `no`) from the JSONL file.
  - Generate a response from the tuned model.
- Accuracy is computed as the ratio of correct predictions to total samples.

This evaluation verifies the effectiveness of fine-tuning by measuring performance improvements on the validation dataset.

---


## Key Metrics

| Stage          | Accuracy |
|----------------|----------|
| Pre-finetune   | 86%     |
| Post-finetune  | 93%   |

> Accuracy is computed by comparing the model's response text (`yes` or `no`) to the ground-truth label.

---


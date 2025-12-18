---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:9
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: I am looking for a COO for my company in China and I want to see
    if they are culturally a right fit for our company. Suggest me an assessment that
    they can complete in about an hour
  sentences:
  - 'Interpersonal Communications Interpersonal Communications: This adaptive test
    measures the candidate''s knowledge of how to employ effective verbal and non-verbal
    communication to send his or her message and manage…'
  - 'Occupational Personality Questionnaire OPQ32r Occupational Personality Questionnaire
    OPQ32r: The SHL Occupational Personality Questionnaire, the OPQ32, is one of the
    most widely used and respected measures of workplace behavioural style in the
    world…'
  - 'Enterprise Leadership Report 2.0 Enterprise Leadership Report 2.0: Assess and
    benchmark your leaders against enterprise leadership - the model for leader impact
    to drive business results in a complex work environment.  For…'
- source_sentence: "Based on the JD below recommend me assessment for the Consultant\
    \ position in my organizations. The assessment should not be more than 90 mins\n\
    b Description\n\n Job Purpose \n\nResponsibilities\n\nThe Consultant role supports\
    \ the broader professional services organization by leading the delivery of impactful\
    \ client solutions and advising on client programs. Consultants are expected to\
    \ deliver solutions, services, and insights that are driven by industry best practices\
    \ and that positively impact client business objectives and partnerships. Individuals\
    \ in this role provide I/O technical guidance to internal and external stakeholders,\
    \ and drive continuous improvement related to project delivery.  Primary Duties\
    \ and Responsibilities  Primary job duties and responsibilities include but may\
    \ not be limited to:\n\n Planning, executing, and documenting projects related\
    \ to clients’ talent assessment and development needs (e.g., job analysis, validation,\
    \ leadership assessment, succession planning, program implementation, adverse\
    \ impact analyses, best practices). \n Leading projects; ensuring overall quality\
    \ for all client deliverables and guiding other to do the same. \n Serving as\
    \ an expert resource to clients and internal stakeholders about issues related\
    \ to talent assessment, selection, and development. \n Linking human capital insights\
    \ to business outcomes and demonstrating ROI. \n Collaborating with members of\
    \ the Sales team to manage existing programs and identify new business opportunities.\
    \ \n Maintaining up-to-date knowledge of the field including empirical research,\
    \ industry trends, and competitors. \n Providing talent analytics and data insights\
    \ to technical and non-technical stakeholders. \n Contributing to the ongoing\
    \ definition and development of SHL’s offerings and methodologies. \n Presenting\
    \ to and interacting with senior client stakeholders. \n Providing thought leadership;\
    \ working with delivery and commercial teams to drive improvements in the scoping\
    \ and execution of projects/deliverables. \n\n Key Competencies: \n\n Decision\
    \ Making \n Collaboration \n Building Relationships \n Creativity and Innovation\
    \ \n Planning and Organizing \n Delivering Results \n Adaptability \n\n Qualifications\
    \ include: \n\n A graduate degree in Industrial/Organizational psychology or a\
    \ related discipline \n 2+ years of relevant work experience in applied talent\
    \ assessment, talent management, or employee selection, including program design\
    \ and implementation \n Client-facing delivery experience with design and implementation\
    \ of talent assessment programs (including job analysis, criterion validation,\
    \ content validation, and interviews) \n Selection system development and / or\
    \ evaluation \n Proficient in at least one statistical program (e.g., SPSS, R)\
    \ \n Strong communication skills and presenting technical results to non-technical\
    \ stakeholders \n\n#IO #IndustrialPsychology #psychtalent #talentmanagement #talentassessment\
    \ #psychometrics #consultant\n\nAbout Us\n\nWe unlock the possibilities of businesses\
    \ through the power of people, science, and technology.\n\nWe started this industry\
    \ of people insight more than 40 years ago and continue to lead the market with\
    \ powerhouse product launches, ground-breaking science, and business transformation.\n\
    \nWhen you inspire and transform people’s lives, you will experience the greatest\
    \ business outcomes possible. SHL’s products insights, experiences, and services\
    \ can help achieve growth at scale.\n\nMore at shl.com.\n\nWhat SHL Can Offer\
    \ You\n\n An inclusive culture. \n A fun and flexible workplace where you’ll be\
    \ inspired to do your best work. \n Employee benefits package that takes care\
    \ of you and your family. \n Support, coaching, and on-the-job development to\
    \ achieve career success. \n The ability to transform workplaces around the world\
    \ for others. \n\nSHL is an equal opportunity employer."
  sentences:
  - 'Interpersonal Communications Interpersonal Communications: This adaptive test
    measures the candidate''s knowledge of how to employ effective verbal and non-verbal
    communication to send his or her message and manage…'
  - 'English Comprehension (New) English Comprehension (New): Multiple-choice test
    that measures vocabulary, grammar and reading comprehension skills.'
  - 'Occupational Personality Questionnaire OPQ32r Occupational Personality Questionnaire
    OPQ32r: The SHL Occupational Personality Questionnaire, the OPQ32, is one of the
    most widely used and respected measures of workplace behavioural style in the
    world…'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Based on the JD below recommend me assessment for the Consultant position in my organizations. The assessment should not be more than 90 mins\nb Description\n\n Job Purpose \n\nResponsibilities\n\nThe Consultant role supports the broader professional services organization by leading the delivery of impactful client solutions and advising on client programs. Consultants are expected to deliver solutions, services, and insights that are driven by industry best practices and that positively impact client business objectives and partnerships. Individuals in this role provide I/O technical guidance to internal and external stakeholders, and drive continuous improvement related to project delivery.  Primary Duties and Responsibilities  Primary job duties and responsibilities include but may not be limited to:\n\n Planning, executing, and documenting projects related to clients’ talent assessment and development needs (e.g., job analysis, validation, leadership assessment, succession planning, program implementation, adverse impact analyses, best practices). \n Leading projects; ensuring overall quality for all client deliverables and guiding other to do the same. \n Serving as an expert resource to clients and internal stakeholders about issues related to talent assessment, selection, and development. \n Linking human capital insights to business outcomes and demonstrating ROI. \n Collaborating with members of the Sales team to manage existing programs and identify new business opportunities. \n Maintaining up-to-date knowledge of the field including empirical research, industry trends, and competitors. \n Providing talent analytics and data insights to technical and non-technical stakeholders. \n Contributing to the ongoing definition and development of SHL’s offerings and methodologies. \n Presenting to and interacting with senior client stakeholders. \n Providing thought leadership; working with delivery and commercial teams to drive improvements in the scoping and execution of projects/deliverables. \n\n Key Competencies: \n\n Decision Making \n Collaboration \n Building Relationships \n Creativity and Innovation \n Planning and Organizing \n Delivering Results \n Adaptability \n\n Qualifications include: \n\n A graduate degree in Industrial/Organizational psychology or a related discipline \n 2+ years of relevant work experience in applied talent assessment, talent management, or employee selection, including program design and implementation \n Client-facing delivery experience with design and implementation of talent assessment programs (including job analysis, criterion validation, content validation, and interviews) \n Selection system development and / or evaluation \n Proficient in at least one statistical program (e.g., SPSS, R) \n Strong communication skills and presenting technical results to non-technical stakeholders \n\n#IO #IndustrialPsychology #psychtalent #talentmanagement #talentassessment #psychometrics #consultant\n\nAbout Us\n\nWe unlock the possibilities of businesses through the power of people, science, and technology.\n\nWe started this industry of people insight more than 40 years ago and continue to lead the market with powerhouse product launches, ground-breaking science, and business transformation.\n\nWhen you inspire and transform people’s lives, you will experience the greatest business outcomes possible. SHL’s products insights, experiences, and services can help achieve growth at scale.\n\nMore at shl.com.\n\nWhat SHL Can Offer You\n\n An inclusive culture. \n A fun and flexible workplace where you’ll be inspired to do your best work. \n Employee benefits package that takes care of you and your family. \n Support, coaching, and on-the-job development to achieve career success. \n The ability to transform workplaces around the world for others. \n\nSHL is an equal opportunity employer.',
    'Occupational Personality Questionnaire OPQ32r Occupational Personality Questionnaire OPQ32r: The SHL Occupational Personality Questionnaire, the OPQ32, is one of the most widely used and respected measures of workplace behavioural style in the world…',
    'English Comprehension (New) English Comprehension (New): Multiple-choice test that measures vocabulary, grammar and reading comprehension skills.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.2337, 0.0574],
#         [0.2337, 1.0000, 0.2227],
#         [0.0574, 0.2227, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 9 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 9 samples:
  |         | sentence_0                                                                          | sentence_1                                                                         |
  |:--------|:------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                             |
  | details | <ul><li>min: 12 tokens</li><li>mean: 82.78 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 27 tokens</li><li>mean: 40.78 tokens</li><li>max: 51 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                         | sentence_1                                                                                                                                                                                                                                        |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>I want to hire new graduates for a sales role in my company, the budget is for about an hour for each test. Give me some options</code>                                                      | <code>Business Communication (adaptive) Business Communication (adaptive): This is an adaptive test that measures knowledge of communicating in the workplace. It measures the skills necessary to communicate effectively with coworkers…</code> |
  | <code>I am looking for a COO for my company in China and I want to see if they are culturally a right fit for our company. Suggest me an assessment that they can complete in about an hour</code> | <code>Enterprise Leadership Report 1.0 Enterprise Leadership Report 1.0: Assess and benchmark your leaders against enterprise leadership - the model for leader impact to drive business results in a complex work environment.  For…</code>      |
  | <code>Content Writer required, expert in English and SEO.</code>                                                                                                                                   | <code>English Comprehension (New) English Comprehension (New): Multiple-choice test that measures vocabulary, grammar and reading comprehension skills.</code>                                                                                    |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Framework Versions
- Python: 3.13.3
- Sentence Transformers: 5.2.0
- Transformers: 4.57.3
- PyTorch: 2.9.1+cpu
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
---
language:
- en
size_categories:
- 1K<n<10K
task_categories:
- text-generation
- question-answering
dataset_info:
  features:
  - name: name
    dtype: string
  - name: description
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: 14924724
    num_examples: 5755
  download_size: 2153926
  dataset_size: 14924724
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: cc-by-4.0
tags:
- roleplay
- characters
---

<h1 align="center"> 🎭 Roleplay TTL</h1>
<p align="center">
    <img src="https://bots-ttl.s3.amazonaws.com/intro1.png" alt="Your Image" width="500">
</p>

<p align="center">Let AI be any characters you want to play with!</p>

## Dataset Overview

This dataset trains conversational AI to embody a wide range of original characters, each with a unique persona. It includes fictional characters, complete with their own backgrounds, core traits, relationships, goals, and distinct speaking styles.

## Dataset Details

- **Curated by:** [Hieu Minh Nguyen](mywebleo.com)
- **Language(s) (NLP):** Primarily English (with potential for multilingual extensions)
- **License:** Creative Commons Attribution 4.0 International License
- **Version:** 1.0 (The new version will be updated soon with topics included for the dataset and 10000+ more entries.)

## Dataset Description

### The dataset includes:
- Name and the description of the character.
- System messages that define each character's persona.
- Conversational exchanges demonstrating typical reactions in various scenarios.
- Coverage of different emotions and topics, with direct quotes and signature linguistic ticks.
- Includes a wide array of characters, ranging from well-known fictional figures to **completely original, self-created personas**.

#### Dataset Composition
- **Number of Rows:** Over 5000 entries, each representing a unique interaction.
- **Interaction Style:** Each dataset entry consists of a system message defining the character's traits, followed by 3-5 conversational exchanges between the character and a user.
  
#### Dataset Goals and Applications
- **Training Objectives:** Ideal for training AI in role-playing applications, chatbots, interactive storytelling, and creative writing tools.
- **Research Value:** Useful for studies in character-driven narrative generation, conversational AI, and creative writing in AI.
- **Out-of-Scope Use:** Not suited for tasks unrelated to conversational or creative AI.

#### Conversational Dynamics
- **Realism in Dialogue:** Each exchange is crafted to mirror realistic conversations, maintaining the authenticity of characters' voices.
- **Language Variability:** Diverse linguistic styles and dialects are used, tailored to each character's background and persona.
- **Humor and Wit:** Includes witty banter and humorous exchanges, adding a layer of entertainment and relatability.

## Dataset Structure

- `name`: Name of the character.
- `description`: Detailed description of the character's persona. 
- `text`: Corresponding responses in the character's unique style.
  
The "text" dataset is formatted as follows (the system message and 4-5 following conversations):
<|system|>...</s>\n<|user|>...</s>\n<|assistant|>...</s>\n<|user|>\n<|assistant|>...</s>

## Data Creation and Processing

Characters are created using imaginative writing of [Gemini Pro](https://deepmind.google/technologies/gemini/#build-with-gemini), ensuring a diverse range of personas. Conversations are scripted to reflect different scenarios, emotions, and interactions.

---
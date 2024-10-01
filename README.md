# Hallyuminus.AI

[Illuminus.ai] **Dialogue System for screenwriting K-drama.**

- This approach targets on using **open-sourced sSsM** (several Small-sized Model), not all-in-1 closed-sourced lLLM (lucirous Large Language Model, like ChatGPT, Gemini, etc.), to simulate participants in a dialogue, including the **narrator** and **characters**.

**Note:** 
- [Character.AI](https://character.ai/) seems very capable of imitating character's characteristics. However, in my experience, it is very easy to manipulate the conversation with the trained bots, so I feel bored in a very short time.
- [Dialogue Model](https://huggingface.co/microsoft/GODEL-v1_1-base-seq2seq) is NOT capable of responding as a participant in a dialogue. It should be named as **Answering Model**. Hence, the result from these pretrained QnaModel doesn't sound good.
--------------------------
## To-Do List

- [x] GUI
- [x] **Character** Builder
- [x] **Relation** Builder
- [x] **Context** Builder
- [x] **Memory** Manager (naïve)
- [ ] **Emotion** Manager
- [x] **Dialogue** Manager
- [ ] **Language** Manager

--------------------------
## Future Improvements

- [ ] **Emotion**-triggered **Memory** Manager (RAG + Graph)
- [ ] Train **LoRA** as **Character** and **Context** Adaptation
- [ ] Finetune **MLM** for customized tasks
- [ ] Voice Dialogue (T2S, S2T)

--------------------------
## Dataset

--------------------------
## User Guide

--------------------------
## References
[Story Generation](https://github.com/yingpengma/Awesome-Story-Generation)

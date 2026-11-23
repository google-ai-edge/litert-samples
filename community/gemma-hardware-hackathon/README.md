# Google Gemma Hardware Hackathon (AGI House)

## 1. About the Hackathon

On **October 24, 2026**, **Google DeepMind** and **[AGI House](https://www.agihouse.org/events/the-google-gemma-hardware-hackathon)** co-hosted the **Google Gemma Hardware Hackathon** in Hillsborough, California. Over 100 builders, engineers, and hardware hackers gathered for a one-day build sprint to bring **Gemma 4** to real physical hardware under a single constraint: **100% on-device execution with no cloud calls**.

Teams built across a wide range of edge platforms—including Android/Pixel phones, iOS devices, Raspberry Pi 5, NVIDIA Jetson (Orin Nano & AGX Thor), Qualcomm, Arduino, and humanoid/arm robots—submitting 40+ working prototypes by the end of the day. Many teams powered their on-device inference with **[LiteRT](https://github.com/google-ai-edge/LiteRT)** and **[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)**.

---

## 2. Best Use of LiteRT Awards

Three projects were recognized in the **Best Use of LiteRT** track for showcasing how on-device ML and LLMs unlock offline, private, and real-time experiences:

### 🥇 1st Place — [Sherpa Survival](https://github.com/KulkarniAnirudh26/gemini-hackathon-survival-app)

* **Team:** An Sung, Anirudh Kulkarni, Chengjiang Long, Grace Huang
* **Demo:** [Demo Video](https://youtube.com/shorts/p-rF7Cr_h_8) · [Event Showcase](https://youtu.be/SvFT9uPeQ8s)
* **Runtime:** LiteRT (`2.3.0`) + LiteRT-LM (Android & iOS, Flutter)
* **Overview:** An offline hiking and survival assistant that pairs local GPS trail routing and BM25 survival-guide retrieval with a four-model native on-device pipeline—Zipformer CTC (ASR), Laya Multilingual (intent routing), Gemma 4 E2B-it (multimodal reasoning), and KittenTTS Nano (TTS)—running completely without cell service.

### 🥈 2nd Place — [DJemma](https://github.com/ddbrx/gemma4-hack-djemma)

* **Team:** Dima Dobrynin
* **Demo:** [Demo Video](https://youtu.be/BJOgSw6m3HI) · [Event Showcase](https://youtu.be/NcihV_lpUmM)
* **Runtime:** LiteRT-LM (`0.18.1`) + MediaPipe Face Landmarker (Android)
* **Overview:** A hands-free Android music-making app designed for accessibility that lets users play melodies and control tempo using head movements and facial expressions, while Gemma 4 E2B runs locally on-device to adapt the backing track, interpret mood photos, and assist with head-typing.

### 🥉 3rd Place — [Wonder Agent](https://github.com/frankdu/hack_android_base)

* **Team:** Frank Du
* **Demo:** [Demo Video](https://www.youtube.com/shorts/5o3rZJnITXU) · [Event Showcase](https://youtu.be/fzYKXhpegh4)
* **Runtime:** LiteRT (`2.3.0`) + LiteRT-LM (`0.18.0`) (Android)
* **Overview:** An on-device language education app that helps children learn English vocabulary through pictures, voice, and interactive conversation, running Gemma 4 (E2B/E4B) and local speech/vision pipelines entirely on the phone after setup.

---

## 3. Community Projects Using LiteRT & LiteRT-LM

The table below catalogs **29 publicly accessible hackathon projects** whose repositories were verified to use **LiteRT** and/or **LiteRT-LM** in code (repositories that were private/unreachable or did not contain LiteRT/LiteRT-LM code are omitted).

| Project | Code | Demo | Runtime | Summary |
| :--- | :---: | :---: | :--- | :--- |
| **Sherpa Survival** 🥇 *Best Use of LiteRT (1st)* | [GitHub](https://github.com/KulkarniAnirudh26/gemini-hackathon-survival-app) | [Video](https://youtube.com/shorts/p-rF7Cr_h_8) | LiteRT + LiteRT-LM | Offline Flutter hiking assistant combining local GPS trail routing with a four-model LiteRT/LiteRT-LM speech, routing, and Gemma 4 vision-language pipeline. |
| **DJemma** 🥈 *Best Use of LiteRT (2nd)* | [GitHub](https://github.com/ddbrx/gemma4-hack-djemma) | [Video](https://youtu.be/BJOgSw6m3HI) | LiteRT-LM | Hands-free Android music app controlled via head pose and facial expressions, with on-device Gemma 4 E2B adapting the backing track in real time. |
| **Wonder Agent** 🥉 *Best Use of LiteRT (3rd)* | [GitHub](https://github.com/frankdu/hack_android_base) | [Video](https://www.youtube.com/shorts/5o3rZJnITXU) | LiteRT + LiteRT-LM | Offline Android language-learning app for children that connects words, illustrations, voice, and interactive conversation using on-device Gemma 4. |
| **SwarmMind (Robots with Redundant Inference)** ⭐ *NVIDIA's Choice* | [GitHub](https://github.com/kevinhaoaus/gemmahackathon) | [Video](https://youtu.be/xQsxAvnHYsE) | LiteRT + LiteRT-LM | Fault-tolerant local multi-device Gemma 4 swarm with mid-generation failover for a lab robot arm, using LiteRT-LM and `ai-edge-litert` on Raspberry Pi 5. |
| **Advo** | [GitHub](https://github.com/Jennifer-xxx/Advo) | [Video](https://drive.google.com/file/d/1q2mWjmGhZkMilqcaVABhJTeqLcN9Ir_P/view?usp=sharing) | LiteRT-LM | Private on-device Android health companion that turns symptoms into appointment briefs and helps patients understand medical conversations offline. |
| **Anu** | [GitHub](https://github.com/sandeepvijayarao09/anu-gemma-hackathon) | — | LiteRT + LiteRT-LM | Android personal assistant with an on-device memory layer running Gemma 4, EmbeddingGemma, and Qualcomm NPU vision detection locally. |
| **Chora-PersonalOS** | [GitHub](https://github.com/louisazz/Chora-PersonalOS-2026-DeepMind-AGIHouse-Hackathon) | [Web](https://chora-life-in-continuity.louisazhang22.chatgpt.site) | LiteRT-LM | Continuous personal AI OS running Gemma 4 E2B/E4B and EmbeddingGemma 2 directly in the browser via `@litert-lm/core` WebGPU. |
| **Cross-Embodiment Gemma Harness** | [GitHub](https://github.com/MoissanClub/gemma-hackathon) | [Video](https://youtube.com/shorts/F73ZXOS05a0) | LiteRT-LM | Cross-embodiment local robotics harness with LiteRT-LM GPU/CPU audio and vision inference probes for Gemma 4 E2B. |
| **DejaLango** | [GitHub](https://github.com/Wenjix/DejaLango) | [Drive](https://drive.google.com/drive/folders/1L4qC7gBEsvmhoEndhSELoo-nAqdrPqje?usp=sharing) | LiteRT-LM | Real-time conversational language coach on Pixel + smart glasses powered by Gemma 4 E2B and EmbeddingGemma 2 on Google Tensor NPU. |
| **EdgeHand** | [GitHub](https://github.com/nicolewongbiz/gemmahackathon) | [Slides](https://claude.ai/artifact/4p3tHhMQAsZiXFErTiHqBo#slide-c73fe23465bb) | LiteRT-LM | Offline phase-aware robotic surgical scrub nurse using multimodal Gemma 4 E4B via LiteRT-LM to predict and hand off instruments. |
| **Elsewhere** | [GitHub](https://github.com/rrwwee/productivity) | [Web](https://rrwwee.github.io/productivity/) | LiteRT-LM | On-device Android focus companion that inspects screen context with Gemma 4 E2B on GPU to detect doomscrolling and suggest alternatives. |
| **Family Petoi** | [GitHub](https://github.com/kiddjacky/family_petoi) | — | LiteRT-LM | Interactive parent–child robotic toy experience on Android + Petoi robots using LiteRT-LM for Gemma 4 voice chat and local embeddings. |
| **Fit Check** | [GitHub](https://github.com/sbonab/fit-check-gemma-hackathon-2026) | [Pitch PDF](https://github.com/sbonab/fit-check-gemma-hackathon-2026/blob/main/output/pdf/fit-check-pitch.pdf) | LiteRT-LM | Playful Android outfit-roast and styling companion powered entirely on-device by Gemma 4 E2B vision via LiteRT-LM. |
| **gem-mesh** | [GitHub](https://github.com/demobo-com/gem-mesh) | [Web](https://local-ai-mesh.onrender.com/) | LiteRT-LM | Open private AI compute mesh uniting phones, laptops, and local GPUs with Android LiteRT-LM clients and Python LiteRT-LM mesh nodes. |
| **gemma-tiered** | [GitHub](https://github.com/whatdhack/gemma-tiered) | [GIF](https://github.com/whatdhack/gemma-tiered/blob/main/docs/demo.gif) | LiteRT-LM | System 1 / System 2 tiered robot controller using FunctionGemma 270M via LiteRT-LM for low-latency reflex tool calling. |
| **Gemmedic** | [GitHub](https://github.com/michael-L-i/robot-circuit) | [Video](https://drive.google.com/file/d/1FMjhSUzffjARLnGSzxEm9JTm_Rt-2S8Z/view?usp=sharing) | LiteRT-LM | Autonomous on-device circuit-repair engineer using Gemma 4 E4B vision via LiteRT-LM to diagnose and repair hardware with a robot arm. |
| **Gremma: A memory you can ask** | [GitHub](https://github.com/ajjpatel/gremma) | [Video](https://github.com/ajjpatel/gremma/blob/main/RPReplay_Final1792893114.MP4) | LiteRT-LM | Ambient on-device AI caretaker for grandparents using LiteRT-LM for Gemma 4 conversation and EmbeddingGemma 2 memory retrieval. |
| **Mediator** | [GitHub](https://github.com/theGlenn/mediator-ai) | — | LiteRT-LM | Local Android conflict-mediation app where two speakers take timed turns and Gemma 4 E2B builds a shared neutral summary offline. |
| **Mise** | [GitHub](https://github.com/neelsirivara/mise-gemma) | [Drive](https://drive.google.com/drive/folders/1sZX-b4W-rNs2RJf6p4tF0rCZEamXeW19?usp=sharing) | LiteRT-LM | Multimodal cooking assistant that inspects your fridge and guides recipes using Gemma 4 E2B and EmbeddingGemma 2 on LiteRT-LM. |
| **money counter (`rax-gemma`)** | [GitHub](https://github.com/arminforoughi/rax-gemma) | [Loom](https://www.loom.com/share/9d40a0d6854f46efb5b5585a420b0ab4) | LiteRT-LM | General-purpose robotic object manipulation running vision-guided Gemma 4 E2B directly in-process on a Raspberry Pi 5 via LiteRT-LM. |
| **Plush Intelligence** | [GitHub](https://github.com/patrick-hanley/sentient) | [Slides](https://docs.google.com/presentation/d/1YX6aYuk6CPzpZ-Sm3XTMDGyObIT5EDh6/edit?usp=sharing) | LiteRT-LM | Interactive plush toys that converse with children and each other using audio-native Gemma 4 E2B on LiteRT-LM and local `.tflite` vision models. |
| **ProductiBot** | [GitHub](https://github.com/Maltomatic/gemma_hackathon) | [Slides](https://claude.ai/artifact/LwjGGKKsNwtc4Uw9VUKVa4#slide-3dd33bdac6c2) | LiteRT-LM | Posture-monitoring companion robot pairing an Android Gemma 4 vision judge on LiteRT-LM with a humanoid robot action planner. |
| **Pupper Companion** | [GitHub](https://github.com/devesh-aggarwal/pupper-hackathon) | — | LiteRT | Proactive on-device companion robot using `ai-edge-litert` to run YAMNet environmental sound classification locally. |
| **Remy** | [GitHub](https://github.com/yunhaiee/Remy) | [Video](https://youtu.be/qm3rRvegRzY) | LiteRT-LM | Hands-free iOS cooking coach running Gemma 4 E2B via the LiteRT-LM Swift SDK alongside real-time on-device vision. |
| **robotachi** | [GitHub](https://github.com/AdamBangash/agi-hackathon) | [Video](https://www.youtube.com/shorts/yvjZz1CgJmQ) | LiteRT-LM | Real-time Unitree G1 humanoid mimicry and vision-driven tool calling backed by a local LiteRT-LM Gemma 4 E2B server on Jetson. |
| **Speechly** | [GitHub](https://github.com/HelloJessieee/speechly) | [Web](https://speechlyweb.netlify.app/) | LiteRT-LM | Personal pronunciation coach for deaf and hard-of-hearing speakers powered by on-device Gemma 4 E2B via `litert-lm-api`. |
| **Travis (Backtrail)** | [GitHub](https://github.com/sdmad737/backtrail) | [Video](https://drive.google.com/file/d/14rSfdvHrqlo4mbRNwhnvft8HFyQ8QajA/view?usp=sharing) | LiteRT-LM | Offline visual-recall trail navigation Android app running Gemma 4 E2B and EmbeddingGemma 2 on Google Tensor G6 NPU via LiteRT-LM. |
| **Wayfinder** | [GitHub](https://github.com/bouncypitch/wayfinder) | [Video](https://github.com/bouncypitch/wayfinder/releases/download/v1.0-hackathon/wayfinder_demo_video.mp4) | LiteRT-LM | Indoor digital-twin mapping and humanoid wayfinding system using LiteRT-LM on Android for local RGB-D object labeling. |
| **Your Private Embedded Gemma Mesh** | [GitHub](https://github.com/ubopod/ubo_app/compare/main...smart-switcher) | [Video](https://drive.google.com/file/d/1ipDvEZXTvaS9Sh51JdRiuT03baGI6m9M/view?usp=sharing) | LiteRT-LM | Personal domain-aware agent router on Raspberry Pi 5 using LiteRT-LM with EmbeddingGemma 2 and Gemma 4 E2B/12B across a private device mesh. |

---

*Built something with LiteRT or LiteRT-LM at the Google Gemma Hardware Hackathon and don't see your project listed (or recently made your repository public)? Feel free to open a pull request to add it!*

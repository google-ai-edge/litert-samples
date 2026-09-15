# MiniCPM

Model recipes, conversion scripts, instructions, and utilities for OpenBMB's MiniCPM family on [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM), the LLM runtime of [LiteRT](https://github.com/google-ai-edge/litert).

* [`minicpm5_2b/`](minicpm5_2b/) — MiniCPM5-2B (2.5B dense, hybrid thinking): how to run the published bundles from the command line, Python, Android and iOS, with the thinking switch and the measured speeds; [`converted/`](minicpm5_2b/converted/) has the conversion that runs on both the CPU and GPU backends, with the two post-export steps that make the difference written down and measured.

# litert-community trending page

A static page that lists the trending and the recently created models of [litert-community](https://huggingface.co/litert-community) on Hugging Face, the picks on its organization card, the samples in this repository that run these models, and the Spaces of the org's Demos collection. The two model lists show each model's formats: `.tflite` files for [LiteRT](https://github.com/google-ai-edge/litert), `.litertlm` bundles for [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM), and `.task` files.

It reads four sources when it loads: the org's model list from the Hugging Face API, the organization card, the sample tables of [`samples/litert/README.md`](../../samples/litert/README.md) and [`samples/litert_lm/README.md`](../../samples/litert_lm/README.md) from this repository's GitHub Pages, and the org's [Demos collection](https://huggingface.co/collections/litert-community/demos-68efdb213c93efee10c2116f). There is no data file in this directory to update. When the card has no heading with the word "pick" or "picks", or a sample table no Sample and Model columns, that part of the page says so and the other parts still load.

- [`space/`](space/): the page, `index.html`, `app.js`, `style.css` and the `README.md` card, the four files a Space needs.
- [`test/`](test/): the checks, which stay out of a Space; [`test/README.md`](test/README.md) lists the three commands.

To look at the page on your machine, run the line below and open `http://localhost:8000/` (every source allows cross-origin requests, so any local server works):

```bash
cd space && python3 -m http.server 8000
```

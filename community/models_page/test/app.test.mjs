// node --test   (from the directory above; nothing here goes to the Space)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
    nextLink, normalizeModels, indexBoard, joinRows, filterModels, sortModels, taskCounts, deviceOptions,
    matchingRows, hasBenchmarks, cardSummary, loadModels, esc, fmt, fmtBytes, sourceModels, parseRecipes, joinRecipes,
    familyOf, sizeOf, bucketOf, familyCounts, sizeCounts, commandLines, SIZE_BUCKETS, SIZE_NONE,
} from '../space/app.js';

const fixture = name => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url)));
const recipesText = () => readFileSync(new URL('./fixtures/recipes.md', import.meta.url), 'utf8');
const NONE = { task: '', family: '', bucket: '', format: '', benchmarks: false, recipe: false, platform: '', device: '', accelerator: '', q: '' };
function joined() {
    const models = normalizeModels(fixture('models.json'));
    const index = indexBoard(fixture('board.json'));
    return { models, index, ...joinRows(models, index) };
}
const names = list => list.map(m => m.name);
const find = (models, name) => models.find(m => m.name === name);

test('normalizeModels reads task, files, formats, the gated flag and the source model, and leaves out a repo without files', () => {
    const { models } = joined();
    assert.equal(models.length, 10);
    assert.equal(find(models, 'Test'), undefined);
    assert.deepEqual(find(models, 'MiniCPM5-2B').source, ['openbmb/MiniCPM5-2B']);
    assert.deepEqual(find(models, 'Qwen3-0.6B').source, ['Qwen/Qwen3-0.6B']);
    assert.deepEqual(find(models, 'MobileNet-v2').source, [], 'a card without base_model');
    assert.equal(find(models, 'whisper-acft').source.length, 6, 'a list keeps every checkpoint');
    assert.equal(find(models, 'MiniCPM5-2B').recipe, null, 'no recipe before the join');
    assert.equal(find(models, 'MobileNet-v2').task, 'image-classification');
    assert.deepEqual(find(models, 'MobileNet-v2').formats, ['tflite']);
    assert.ok(find(models, 'MobileNet-v2').files.tflite.includes('mobilenet_v2.tflite'));
    assert.ok(find(models, 'Qwen3-0.6B').formats.includes('litertlm'));
    assert.equal(find(models, 'Gemma3-1B-IT').gated, true);
    assert.equal(find(models, 'gemma-4-E2B-it-litert-lm').task, null);
    assert.deepEqual(find(models, 'Gemma3-4B-IT').formats, ['task']);
    assert.ok(find(models, 'whisper-acft').files.tflite.some(n => n.includes('/')), 'files in folders keep their path');
    assert.match(find(models, 'U-2-Net').updated, /^\d{4}-\d{2}-\d{2}$/);
});

test('normalizeModels survives entries with missing or odd fields', () => {
    const out = normalizeModels([
        { id: 'litert-community/a' }, { id: 'litert-community/b', private: true }, null, { modelId: 'x' }, 7,
        { id: 'litert-community/c', pipeline_tag: 7, siblings: 'none', gated: 'auto', downloads: 'many', lastModified: null },
        { id: 'litert-community/d', siblings: [null, {}, { rfilename: 3 }, { rfilename: 'a.TFLITE' }] },
    ]);
    assert.deepEqual(out.map(m => m.id), ['litert-community/a', 'litert-community/c', 'litert-community/d']);
    assert.deepEqual(out[0], { id: 'litert-community/a', name: 'a', task: null, downloads: 0, updated: '', gated: false, empty: false,
        files: { tflite: [], litertlm: [], task: [] }, formats: [], source: [], family: { key: 'a', label: 'a' }, size: null, bucket: 'none',
        recipe: null, rows: [], lmRows: [] });
    assert.equal(out[1].task, null);
    assert.equal(out[1].gated, true);
    assert.equal(out[1].downloads, 0);
    assert.deepEqual(out[2].formats, ['tflite']);
    const odd = normalizeModels([{ id: 'litert-community/e', siblings: [{ rfilename: 'a.constructor' }, { rfilename: '__proto__' }, { rfilename: 'x.toString' }] }]);
    assert.deepEqual(odd[0].files, { tflite: [], litertlm: [], task: [] }, 'a file ending in an Object.prototype key is not a format');
    assert.throws(() => normalizeModels({ error: 'Rate limit reached' }), /not a model list/);
});

test('sourceModels keeps repo ids only, once each, at most eight', () => {
    assert.deepEqual(sourceModels({ base_model: 'google/gemma-4-E2B-it' }), ['google/gemma-4-E2B-it']);
    assert.deepEqual(sourceModels({ base_model: ['a/b', 'a/b', 'c/d.e-f_g'] }), ['a/b', 'c/d.e-f_g']);
    assert.deepEqual(sourceModels({ base_model: ['<img src=x onerror=x>', 'javascript:alert(1)', '../../x', 'org/<b>b</b>', 'Qwen2.5-Coder-3B-Instruct', 'a/..', 'a/', '/b', 7, null, 'ok/repo'] }), ['ok/repo']);
    assert.deepEqual(sourceModels({ base_model: Array.from({ length: 12 }, (_, i) => `o/r${i}`) }).length, 8);
    for (const card of [null, undefined, 'x', 7, {}, { base_model: null }, { base_model: 7 }, { base_model: {} }]) assert.deepEqual(sourceModels(card), []);
});

const HOSTILE_RECIPES = `# Model recipes

## Model list

| Recipe | Task | Artifact | Source model | Converted weights |
|---|---|---|---|---|
| [<img src=x onerror=window.__pwned=1>](javascript:alert(1)) | x | x | x | [x](https://huggingface.co/litert-community/MiniCPM5-2B) |
| [\`../../etc/\`](../../etc/) | x | x | x | [x](https://huggingface.co/litert-community/MobileNet-v2) |
| [\`a/b/\`](a/b/) | x | x | x | [<b>x</b>](https://evil.example/litert-community/whisper-tiny) |
| [\`ok/dir/\`](ok/dir/) | x | x | x | [<img src=x onerror=window.__pwned=1>](https://huggingface.co/litert-community/U-2-Net) |
`;

test('parseRecipes reads the model list by its header names and skips a row that names two directories', () => {
    const { recipes, skipped } = parseRecipes(recipesText());
    assert.deepEqual([...recipes.keys()].sort(), ['litert-community/Bonsai-Image-ternary-4B', 'litert-community/MiniCPM5-2B',
        'litert-community/Qwen3-TTS-12Hz-0.6B-Base', 'litert-community/Zipformer-medium-CR-CTC-LiteRT', 'litert-community/wav2vec2-keyword-spotting']);
    assert.deepEqual(recipes.get('litert-community/MiniCPM5-2B'), { path: 'minicpm/minicpm5_2b/', url: 'https://github.com/google-ai-edge/litert-samples/tree/main/models/minicpm/minicpm5_2b/' });
    assert.equal(recipes.get('litert-community/Bonsai-Image-ternary-4B').path, 'bonsai/bonsai_image_4b/');
    assert.equal(skipped, 0, 'the row reserved for two Gemma directories names no repo of the org, so it is not counted');
    const two = parseRecipes('| Recipe | Converted weights |\n|---|---|\n| [x](a/), [y](b/) | [x](https://huggingface.co/litert-community/A) |\n');
    assert.equal(two.recipes.size, 0, 'two directories for one repo give no link');
    assert.equal(two.skipped, 1);
    assert.ok(![...recipes.keys()].some(k => !k.startsWith('litert-community/')), 'weights outside the org are left out');
    const hostile = parseRecipes(HOSTILE_RECIPES);
    assert.deepEqual([...hostile.recipes.entries()], [['litert-community/U-2-Net', { path: 'ok/dir/', url: 'https://github.com/google-ai-edge/litert-samples/tree/main/models/ok/dir/' }]]);
    assert.equal(hostile.skipped, 2, 'the two rows that name a repo of the org but no directory the page accepts');
    const header = '| RECIPE | Converted Weights |\n|---|---|\n';
    assert.equal(parseRecipes(header).recipes.size, 0, 'an empty table is not a failure');
    assert.equal(parseRecipes(header + '| [`a/`](a) | [x](https://huggingface.co/litert-community/A) |\ntext after\n| [`b/`](b) | [x](https://huggingface.co/litert-community/B) |\n').recipes.size, 1, 'the table ends at the first line that is not a row');
    assert.equal(parseRecipes(header + '| [`a/`](a) | [x](https://huggingface.co/litert-community/A/) |\n').recipes.get('litert-community/A').path, 'a/');
    const forms = parseRecipes(header + '| [x](./a/b/) | [x](https://huggingface.co/litert-community/A) |\n'
        + '| [x](https://github.com/google-ai-edge/litert-samples/tree/main/models/c/d/) | [x](https://huggingface.co/litert-community/C) |\n'
        + '| [x](e/f/#run) | [x](https://huggingface.co/litert-community/E) |\n');
    assert.deepEqual([...forms.recipes.values()].map(r => r.path), ['a/b/', 'c/d/', 'e/f/'], 'a ./ prefix, the repo URL and a #fragment are read as the directory');
    assert.equal(forms.skipped, 0);
    const pipe = parseRecipes(header + '| [x](a/b/) | CPU \\| GPU | [x](https://huggingface.co/litert-community/A) |\n| [x](c/d/) | [x](https://huggingface.co/litert-community/C) |\n');
    assert.deepEqual([...pipe.recipes.keys()], ['litert-community/C'], 'a row with another cell count than the header is not read');
    assert.equal(pipe.skipped, 1);
    assert.throws(() => parseRecipes('# no table here\n| Directory | Weights |\n|---|---|\n| a | b |\n'), /no model list/);
    assert.throws(() => parseRecipes(''), /no model list/);
    for (const v of [null, undefined, 7, {}, []]) assert.throws(() => parseRecipes(v), /not text/);
});

test('joinRecipes attaches recipes and reports those of repos the list does not have', () => {
    const { models } = joined();
    const { orphans } = joinRecipes(models, parseRecipes(recipesText()).recipes);
    assert.deepEqual(find(models, 'MiniCPM5-2B').recipe.path, 'minicpm/minicpm5_2b/');
    assert.ok(models.filter(m => m.recipe).length === 1);
    assert.deepEqual(orphans.sort(), ['litert-community/Bonsai-Image-ternary-4B', 'litert-community/Qwen3-TTS-12Hz-0.6B-Base',
        'litert-community/Zipformer-medium-CR-CTC-LiteRT', 'litert-community/wav2vec2-keyword-spotting']);
    assert.deepEqual(names(filterModels(models, { ...NONE, recipe: true })), ['MiniCPM5-2B']);
    joinRecipes(models, new Map());
    assert.equal(find(models, 'MiniCPM5-2B').recipe, null, 'a second join replaces the first');
});

test('indexBoard groups benchmarks by repo and lists only platforms, devices and accelerators that have some', () => {
    const { index } = joined();
    assert.equal(index.byModel.get('litert-community/MobileNet-v2').rows.length, 20);
    assert.equal(index.byModel.get('litert-community/Qwen3-0.6B').lmRows.length, 2);
    assert.deepEqual(index.platforms.map(p => p.id), ['android', 'macos', 'ios']);
    assert.deepEqual(index.accelerators, ['cpu', 'gpu']);
    assert.ok(index.devices.some(d => d.name === 'Pixel 9 Pro' && d.platform === 'android'));
    assert.equal(index.count, 34);
    assert.equal(index.newest, '2026-09-19');
    assert.deepEqual(index.devices.map(d => d.name), ['Galaxy S25 Ultra', 'iPhone 17 Pro', 'Mac Studio (M4 Max)', 'Pixel 9 Pro']);
    const order = index.byModel.get('litert-community/MobileNet-v2').rows.map(r => `${r.platform}/${r.device}/${r.accelerator}/${r.file}`);
    assert.equal(order[0], 'android/Galaxy S25 Ultra/cpu/mobilenet_v2.tflite');
    assert.equal(order[order.length - 1], 'ios/iPhone 17 Pro/gpu/mobilenet_v2.tflite');
});

test('indexBoard throws on a file that is not a board and skips benchmarks it cannot read', () => {
    assert.throws(() => indexBoard({ hello: 'world' }), /no rows/);
    assert.throws(() => indexBoard(null), /no rows/);
    const board = fixture('board.json');
    const [a, b, c, d] = board.rows;
    const index = indexBoard({ rows: [a, { model: 'x' }, null, 5, { ...b, status: 'failed' }, { ...c, latency_ms: null }, { ...d, device: '' }] });
    assert.equal(index.count, 1);
    assert.deepEqual(index.platforms, [{ id: a.platform, name: a.platform }], 'a platform the board does not name still gets an entry');
    assert.equal(indexBoard({ rows: [], generated_at: '2026-09-19T03:08:57Z' }).newest, '2026-09-19');
    assert.equal(indexBoard({ rows: [] }).newest, '');
    assert.equal(indexBoard({ rows: board.rows.map(r => ({ ...r, status: 'ok' })) }).count, 0, 'a renamed status leaves nothing to show');
});

test('joinRows attaches benchmarks and reports those of repos the list does not have', () => {
    const { models, orphans } = joined();
    assert.deepEqual(orphans, []);
    assert.deepEqual(names(models.filter(hasBenchmarks)).sort(), ['Gecko-110m-en', 'MobileNet-v2', 'Qwen3-0.6B', 'whisper-tiny']);
    const board = fixture('board.json');
    board.rows.push({ ...board.rows[0], model: 'someone-else/not-in-the-org' });
    assert.deepEqual(joinRows(normalizeModels(fixture('models.json')), indexBoard(board)).orphans, ['someone-else/not-in-the-org']);
});

test('filterModels: task, format, name, benchmarks', () => {
    const { models } = joined();
    assert.equal(filterModels(models, NONE).length, 10);
    assert.deepEqual(names(filterModels(models, { ...NONE, task: 'image-classification' })), ['MobileNet-v2']);
    assert.deepEqual(names(filterModels(models, { ...NONE, task: '(other)' })), ['gemma-4-E2B-it-litert-lm']);
    assert.ok(filterModels(models, { ...NONE, format: 'litertlm' }).every(m => m.files.litertlm.length > 0));
    assert.deepEqual(names(filterModels(models, { ...NONE, format: 'task' })).includes('Gemma3-4B-IT'), true);
    assert.deepEqual(names(filterModels(models, { ...NONE, q: '  WHISPER-T ' })), ['whisper-tiny']);
    assert.deepEqual(names(filterModels(models, { ...NONE, q: 'litert' })), ['gemma-4-E2B-it-litert-lm'], 'the org prefix is not part of a name');
    assert.deepEqual(filterModels(models, { ...NONE, q: 'community/' }), []);
    assert.equal(filterModels(models, { ...NONE, benchmarks: true }).length, 4);
    assert.deepEqual(filterModels(models, { ...NONE, q: 'no such model' }), []);
});

test('familyOf: the leading letters of the source model, of the repo name without one; sizeOf: the first size token, buckets', () => {
    assert.deepEqual(familyOf('Gemma3-1B-IT', ['google/Gemma-3-1B-IT']), { key: 'gemma', label: 'Gemma' });
    assert.deepEqual(familyOf('gemma-4-E2B-it-litert-lm', ['google/gemma-4-E2B-it']), { key: 'gemma', label: 'gemma' });
    assert.deepEqual(familyOf('MobileNet-v2', []), { key: 'mobilenet', label: 'MobileNet' }, 'the repo name when the card names no source');
    assert.deepEqual(familyOf('whisper-acft', ['openai/whisper-tiny', 'openai/whisper-base']), { key: 'whisper', label: 'whisper' }, 'the first source of a list');
    assert.equal(familyOf('7B-model', []), null, 'a name that starts with a digit has no family');
    assert.equal(sizeOf('Gemma3-1B-IT', []), 1);
    assert.equal(sizeOf('Qwen3-0.6B', []), 0.6);
    assert.equal(sizeOf('SmolLM2-135M-Instruct', []), 0.135, 'M is a thousandth of a B');
    assert.equal(sizeOf('Gecko-110m-en', []), 0.11);
    assert.equal(sizeOf('gemma-4-E2B-it', []), 2, "Gemma's E2B");
    assert.equal(sizeOf('gemma_3_270m_it', []), 0.27, 'underscores separate too');
    assert.equal(sizeOf('phi-4-mini', ['microsoft/Phi-4-mini-instruct']), null, 'a version number is not a size');
    assert.equal(sizeOf('some-model', ['org/Some-Model-3.8B']), 3.8, 'the source model name when the repo name has none');
    assert.equal(sizeOf('yolov8m', []), null, 'a letter before the digits is not a size');
    assert.equal(sizeOf('Qwen3-4bit', []), null, 'a letter after the unit is not a size');
    assert.equal(sizeOf('Llama-1B-4bit', []), 1, 'the first token wins');
    assert.equal(sizeOf('efficientnet_b1', []), null);
    assert.equal(sizeOf('U-2-Net', ['xuebinqin/U-2-Net']), null);
    assert.equal(bucketOf(0.6), 'upto1b');
    assert.equal(bucketOf(1), 'upto1b', 'the top bound is inclusive');
    assert.equal(bucketOf(1.2), '1to4b');
    assert.equal(bucketOf(4), '1to4b');
    assert.equal(bucketOf(4.5), 'over4b');
    assert.equal(bucketOf(null), SIZE_NONE.id);
    assert.deepEqual(SIZE_BUCKETS.map(b => b.id), ['upto1b', '1to4b', 'over4b']);
});

test('normalizeModels files each model under a family and a size bucket; familyCounts and sizeCounts count them', () => {
    const { models } = joined();
    assert.deepEqual(find(models, 'MiniCPM5-2B').family, { key: 'minicpm', label: 'MiniCPM' });
    assert.equal(find(models, 'MiniCPM5-2B').size, 2);
    assert.equal(find(models, 'MiniCPM5-2B').bucket, '1to4b');
    assert.equal(find(models, 'whisper-acft').bucket, 'none');
    assert.deepEqual(familyCounts(models), [
        ['gemma', 3, 'gemma'], ['whisper', 2, 'whisper'], ['gecko', 1, 'Gecko'], ['minicpm', 1, 'MiniCPM'],
        ['mobilenet', 1, 'MobileNet'], ['qwen', 1, 'Qwen'], ['u', 1, 'U'],
    ], 'by count, the spelling seen most often as the label, ties by key');
    const withOther = familyCounts([...models, ...normalizeModels([{ id: 'litert-community/3d-thing', siblings: [{ rfilename: 'a.tflite' }] }])]);
    assert.deepEqual(withOther[withOther.length - 1], ['(other)', 1, 'Other'], 'repos without a family come last');
    assert.deepEqual(sizeCounts(models), [['upto1b', 3, '≤1B'], ['1to4b', 3, '1–4B'], ['over4b', 0, '>4B'], ['none', 4, 'Size not in the name']]);
    assert.equal(sizeCounts(models).reduce((n, [, c]) => n + c, 0), models.length, 'every model is in one bucket');
});

test('filterModels: family and size bucket', () => {
    const { models } = joined();
    assert.deepEqual(names(filterModels(models, { ...NONE, family: 'gemma' })), ['Gemma3-1B-IT', 'gemma-4-E2B-it-litert-lm', 'Gemma3-4B-IT']);
    assert.deepEqual(names(filterModels(models, { ...NONE, family: 'gemma', bucket: 'upto1b' })), ['Gemma3-1B-IT']);
    assert.deepEqual(filterModels(models, { ...NONE, family: '(other)' }), [], 'every fixture model has a family');
    assert.equal(filterModels(models, { ...NONE, bucket: 'none' }).length, 4);
    assert.equal(filterModels(models, { ...NONE, bucket: 'over4b' }).length, 0);
});

test('commandLines: the CLI forms with one file of each format filled in, the shortest name; none for a .task-only repo or a name the lines cannot carry', () => {
    const { models } = joined();
    const mobilenet = commandLines(find(models, 'MobileNet-v2'));
    assert.equal(mobilenet.length, 1);
    assert.deepEqual(mobilenet[0], { format: 'tflite', file: 'mobilenet_v2.tflite', count: 4, lines: [
        'litert download litert-community/MobileNet-v2 --file "mobilenet_v2.tflite" --output MobileNet-v2',
        'litert benchmark MobileNet-v2/mobilenet_v2.tflite --desktop --cpu',
        'litert run MobileNet-v2/mobilenet_v2.tflite --desktop --cpu',
    ] });
    const qwen = commandLines(find(models, 'Qwen3-0.6B'));
    assert.deepEqual(qwen.map(b => b.format), ['litertlm']);
    assert.deepEqual(qwen[0].lines, [
        'litert download litert-community/Qwen3-0.6B --file "Qwen3-0.6B.litertlm" --output Qwen3-0.6B',
        'litert lm benchmark Qwen3-0.6B/Qwen3-0.6B.litertlm',
        'litert lm run Qwen3-0.6B/Qwen3-0.6B.litertlm --prompt "What is the capital of France?"',
    ]);
    assert.deepEqual(commandLines(find(models, 'Gemma3-4B-IT')), [], '.task files have no CLI form');
    assert.match(commandLines(find(models, 'whisper-acft'))[0].lines[1], /^litert benchmark whisper-acft\/base\/acft_whisper_base_5s_drq\.tflite --desktop --cpu$/, 'a file in a folder keeps its path; of the two shortest names (base/…, tiny/…) the Hub lists base first');
    assert.equal(commandLines(find(models, 'gemma-4-E2B-it-litert-lm'))[0].file, 'gemma-4-E2B-it.litertlm', 'the shortest name is the plain build, not the -gpu one the Hub lists first');
    assert.equal(commandLines(find(models, 'Gemma3-1B-IT'))[0].file, 'gemma3-1b-it-int4.litertlm');
    const odd = normalizeModels([
        { id: 'litert-community/<img src=x>', siblings: [{ rfilename: 'a.tflite' }] },
        { id: 'litert-community/spaced', siblings: [{ rfilename: 'my model.tflite' }, { rfilename: 'ok.tflite' }] },
        { id: 'litert-community/glob', siblings: [{ rfilename: 'x[1].tflite' }] },
        { id: 'litert-community/both', siblings: [{ rfilename: 'a.tflite' }, { rfilename: 'b.litertlm' }] },
    ]);
    assert.deepEqual(commandLines(odd[0]), [], 'a repo id the lines cannot carry');
    assert.equal(commandLines(odd[1])[0].file, 'ok.tflite', 'a file name the lines can carry');
    assert.deepEqual(commandLines(odd[2]), [], 'a glob character in the only file name');
    assert.deepEqual(commandLines(odd[3]).map(b => b.format), ['tflite', 'litertlm'], 'one block per format');
});

test('filterModels: a platform, device or accelerator keeps only models benchmarked there', () => {
    const { models } = joined();
    assert.deepEqual(names(filterModels(models, { ...NONE, platform: 'ios' })), ['MobileNet-v2']);
    const pixelGpu = { ...NONE, device: 'Pixel 9 Pro', accelerator: 'gpu' };
    assert.deepEqual(names(filterModels(models, pixelGpu)).sort(), ['Gecko-110m-en', 'MobileNet-v2', 'Qwen3-0.6B', 'whisper-tiny']);
    const hit = matchingRows(find(models, 'MobileNet-v2'), pixelGpu);
    assert.equal(hit.rows.length, 3);
    assert.ok(hit.rows.every(r => r.device === 'Pixel 9 Pro' && r.accelerator === 'gpu'));
    assert.deepEqual(filterModels(models, { ...NONE, device: 'Pixel 9 Pro', platform: 'macos' }), []);
});

test('cardSummary: the file with the most benchmarks, one line per device, one number per accelerator', () => {
    const { models } = joined();
    const mnv2 = find(models, 'MobileNet-v2');
    const s = cardSummary(mnv2.rows, 'litert');
    assert.equal(s.file, 'mobilenet_v2.tflite');
    assert.deepEqual(s.lines.map(l => l.device), ['Galaxy S25 Ultra', 'Pixel 9 Pro', 'Mac Studio (M4 Max)']);
    assert.deepEqual(s.lines[1], { device: 'Pixel 9 Pro', date: '2026-09-18', cells: [{ accelerator: 'cpu', value: 9.847 }, { accelerator: 'gpu', value: 6.409 }] });
    assert.equal(s.shown, 6);
    const gpuOnly = cardSummary(matchingRows(mnv2, { ...NONE, device: 'Pixel 9 Pro', accelerator: 'gpu' }).rows, 'litert');
    assert.deepEqual(gpuOnly.lines.map(l => l.cells.length), [1]);
    const lm = cardSummary(find(models, 'Qwen3-0.6B').lmRows, 'lm');
    assert.deepEqual(lm.tokens, { prefill: 1024, decode: 256 });
    assert.deepEqual(lm.lines[0].cells.map(c => c.value), [3.343, 21.88]);
    assert.equal(lm.shown, 2);
    assert.equal(cardSummary([], 'litert'), null);
});

test('cardSummary: repeats keep the newest number, two devices of one name stay apart, odd conditions add no words', () => {
    const { models } = joined();
    const [cpu, gpu] = find(models, 'Qwen3-0.6B').lmRows;
    const repeat = { ...cpu, date: '2026-09-20', metrics: { ...cpu.metrics, decode_tok_s: 4.5 } };
    const again = cardSummary([cpu, gpu, repeat], 'lm');
    assert.deepEqual(again.lines, [{ device: 'Pixel 9 Pro', date: '2026-09-20', cells: [{ accelerator: 'cpu', value: 4.5 }, { accelerator: 'gpu', value: 21.88 }] }]);
    assert.equal(again.shown, 2);
    const twin = { ...cpu, device_id: 'caiman-36', os: 'Android 16 (API 36)' };
    assert.deepEqual(cardSummary([cpu, twin], 'lm').lines.map(l => l.device), ['Pixel 9 Pro, Android 15 (API 35)', 'Pixel 9 Pro, Android 16 (API 36)']);
    const longer = { ...cpu, conditions: { ...cpu.conditions, max_num_tokens: 4096 } };
    assert.equal(cardSummary([cpu, gpu, longer], 'lm').shown, 2, 'another max_num_tokens is another condition');
    assert.equal(cardSummary([{ ...cpu, conditions: {} }], 'lm').tokens, null);
    assert.equal(cardSummary([{ ...cpu, conditions: null }], 'lm').tokens, null);
    assert.equal(cardSummary([{ ...cpu, conditions: { prefill_tokens: '1024', decode_tokens: 256 } }], 'lm').tokens, null);
});

test('sortModels: benchmarks first, downloads, updated, name', () => {
    const { models } = joined();
    const first = sortModels(models, 'benchmarks');
    assert.ok(first.slice(0, 4).every(hasBenchmarks) && !first.slice(4).some(hasBenchmarks));
    const byDownloads = sortModels(models, 'downloads').map(m => m.downloads);
    assert.deepEqual(byDownloads, [...byDownloads].sort((a, b) => b - a));
    const byDate = sortModels(models, 'updated').map(m => m.updated);
    assert.deepEqual(byDate, [...byDate].sort().reverse());
    assert.equal(sortModels(models, 'name')[0].name, 'Gecko-110m-en');
    assert.equal(models[0].name, 'MobileNet-v2', 'the input keeps its order');
});

test('taskCounts puts repos without a tag last; deviceOptions follows the platform', () => {
    const { models, index } = joined();
    const counts = taskCounts(models);
    assert.equal(counts[counts.length - 1][0], '(other)');
    assert.equal(counts.reduce((n, [, c]) => n + c, 0), 10);
    assert.deepEqual(deviceOptions(index, 'ios'), ['iPhone 17 Pro']);
    assert.equal(deviceOptions(index, '').length, 4);
});

test('nextLink and loadModels follow the Hub paging header and stop at its end', async () => {
    assert.equal(nextLink('<https://huggingface.co/api/models?cursor=abc>; rel="next"'), 'https://huggingface.co/api/models?cursor=abc');
    assert.equal(nextLink('<https://x/prev>; rel="prev", <https://x/next>; rel="next"'), 'https://x/next');
    assert.equal(nextLink(null), null);
    const all = fixture('models.json');
    const calls = [];
    const get = async url => {
        calls.push(url);
        const page = calls.length - 1;
        return { data: all.slice(page * 3, page * 3 + 3), link: (page + 1) * 3 < all.length ? `<https://hub.test/page${page + 1}>; rel="next"` : null };
    };
    assert.equal((await loadModels(get)).length, 10);
    assert.equal(calls.length, 4);
    await assert.rejects(loadModels(async () => ({ data: { error: 'Rate limit reached' }, link: null })), /not a model list/);
    await assert.rejects(loadModels(async () => ({ data: [], link: null })), /came back empty/);
    await assert.rejects(loadModels(async () => { throw new Error('HTTP 429'); }), /HTTP 429/);
    let pages = 0;
    await loadModels(async () => ({ data: [all[0]], link: `<https://hub.test/again${++pages}>; rel="next"` }));
    assert.equal(pages, 20, 'a listing that never ends is cut off');
});

test('esc and the number formats', () => {
    assert.equal(esc('<img src=x onerror="a(\'b\')">&'), '&lt;img src=x onerror=&quot;a(&#39;b&#39;)&quot;&gt;&amp;');
    assert.equal(esc(null), '');
    assert.equal(fmt(42.989), '42.99');
    assert.equal(fmt(21.88, 1), '21.9');
    for (const v of [null, undefined, '', 'abc', NaN, Infinity, {}]) assert.equal(fmt(v), '—');
    assert.equal(fmtBytes(13975904), '14.0 MB');
    assert.equal(fmtBytes(4010670890), '4.01 GB');
    assert.equal(fmtBytes(4507), '5 kB');
    assert.equal(fmtBytes(undefined), '—');
});

// node --test test/app.test.mjs   (from the directory above; nothing here goes to the Space)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
    nextLink, normalizeModels, trendingModels, recentModels, parsePicks, safeLink, inlineHtml, plainText, withoutHtml, parseSamples,
    samplesByModel, samplesFor, parseDemos, loadModels, loadSamples, sourceModels, esc, fmtCount, LIST_LENGTH, MORE_LENGTH, MODELS_URL,
    CARD_URL, SAMPLE_INDEXES, DEMOS_URL,
} from '../space/app.js';

const file = path => readFileSync(new URL(path, import.meta.url), 'utf8');
const read = name => file(`./fixtures/${name}`);
const fixture = name => JSON.parse(read(name));
const models = () => normalizeModels(fixture('models.json'));
const names = list => list.map(m => m.name);
const find = (list, name) => list.find(m => m.name === name);
const mds = picks => picks.items.map(i => i.md);
const ORG = 'https://huggingface.co/litert-community/';
const TREE = 'https://github.com/google-ai-edge/litert-samples/tree/main/';

test('normalizeModels keeps the order of the listing, reads task, formats, counts, the gated flag and the source model, and leaves out a repo without files', () => {
    const list = models();
    assert.equal(list.length, 46);
    assert.equal(find(list, 'Test'), undefined);
    assert.deepEqual(names(list).slice(0, 3), ['gemma-4-E2B-it-litert-lm', 'Gemma3-1B-IT', 'Fun-ASR-Nano-2512']);
    const gemma = find(list, 'gemma-4-E2B-it-litert-lm');
    assert.deepEqual({ ...gemma }, {
        id: 'litert-community/gemma-4-E2B-it-litert-lm', name: 'gemma-4-E2B-it-litert-lm', task: null, likes: 459, downloads: 1398895, trending: 11,
        created: '2026-03-25T23:59:22.000Z', gated: false, empty: false, formats: ['litertlm', 'task'], source: ['google/gemma-4-E2B-it'],
    });
    assert.equal(find(list, 'Gemma3-1B-IT').gated, true);
    assert.equal(find(list, 'Fun-ASR-Nano-2512').task, 'automatic-speech-recognition');
    assert.deepEqual(find(list, 'MobileNet-v2').source, [], 'a card without base_model');
    assert.equal(find(list, 'whisper-acft').source.length, 6, 'a list keeps every checkpoint');
    assert.deepEqual(find(list, 'GLiNER2.5-Decide-LiteRT').formats, ['tflite'], 'files in folders count');
    assert.equal(find(list, 'MiniCPM5-2B').trending, 0.5);
    assert.equal(find(list, 'Kev-4B-LiteRT').trending, 0);
});

test('normalizeModels survives entries with missing or odd fields; a count that is missing is zero, a score that is missing is null', () => {
    const out = normalizeModels([
        { id: 'litert-community/a' }, { id: 'litert-community/b', private: true }, null, { modelId: 'x' }, 7,
        { id: 'litert-community/c', pipeline_tag: 7, siblings: 'none', gated: 'auto', downloads: 'many', likes: -3, trendingScore: -1, createdAt: 5 },
        { id: 'litert-community/d', siblings: [null, {}, { rfilename: 3 }, { rfilename: 'a.TFLITE' }, { rfilename: 'x.constructor' }, { rfilename: 'task' }, { rfilename: 'folder/litertlm' }] },
        { id: 'litert-community/<img src=x>' }, { id: 'no-slash' }, { id: 'litert-community/a\uD800b' }, { id: 'a/b/c' },
    ]);
    assert.deepEqual(out.map(m => m.id), ['litert-community/a', 'litert-community/c', 'litert-community/d'], 'what is not a repo id is left out');
    assert.deepEqual(out[0], { id: 'litert-community/a', name: 'a', task: null, likes: 0, downloads: 0, trending: null, created: '', gated: false, empty: false, formats: [], source: [] });
    assert.deepEqual([out[1].task, out[1].gated, out[1].downloads, out[1].likes, out[1].trending, out[1].created], [null, true, 0, 0, null, '']);
    assert.deepEqual(out[2].formats, ['tflite'], 'a format is the part after the last dot: a file named task is not a .task file');
    assert.throws(() => normalizeModels({ error: 'Rate limit reached' }), /not a model list/);
});

test('sourceModels keeps repo ids only, once each, at most eight', () => {
    assert.deepEqual(sourceModels({ base_model: 'google/gemma-4-E2B-it' }), ['google/gemma-4-E2B-it']);
    assert.deepEqual(sourceModels({ base_model: ['a/b', 'a/b', 'c/d.e-f_g'] }), ['a/b', 'c/d.e-f_g']);
    assert.deepEqual(sourceModels({ base_model: ['<img src=x onerror=x>', 'javascript:alert(1)', '../../x', 'org/<b>b</b>', 'a/..', 7, null, 'ok/repo'] }), ['ok/repo']);
    assert.equal(sourceModels({ base_model: Array.from({ length: 12 }, (_, i) => `o/r${i}`) }).length, 8);
    for (const card of [null, undefined, 'x', 7, {}, { base_model: null }, { base_model: {} }]) assert.deepEqual(sourceModels(card), []);
});

test('trendingModels: the models of the listing whose trending score is above zero, in the listing\'s order; no score at all is an error', () => {
    const list = models();
    const trending = trendingModels(list);
    assert.deepEqual([LIST_LENGTH, MORE_LENGTH], [12, 24]);
    assert.deepEqual(names(trending).slice(0, LIST_LENGTH), ['gemma-4-E2B-it-litert-lm', 'Gemma3-1B-IT', 'Fun-ASR-Nano-2512', 'functiongemma-270m-ft-mobile-actions',
        'gemma-4-12B-it-litert-lm', 'embeddinggemma-300m', 'Bonsai-Image-ternary-4B', 'gemma-4-E4B-it-litert-lm', 'GLiNER2.5-Small-LiteRT', 'laya-LiteRT',
        'Laya-Multilingual-LiteRT', 'GLiNER2.5-Decide-LiteRT']);
    assert.equal(trending.length, 19, 'every model with a score above zero');
    assert.equal(trending.at(-1).name, 'MiniCPM5-2B');
    assert.equal(trendingModels(list, 5).length, 5);
    assert.deepEqual(trendingModels(list.map(m => ({ ...m, trending: 0 }))), [], 'none trending gives an empty list');
    assert.deepEqual(trendingModels([]), []);
    const shuffled = [list[5], list[0], list[2]];
    assert.deepEqual(names(trendingModels(shuffled)), names(shuffled), 'the page does not reorder what the Hub sent');
    assert.throws(() => trendingModels(list.map(m => ({ ...m, trending: null }))), /carries no trending score/);
    assert.equal(trendingModels([{ ...list[0], trending: null }, list[1]]).length, 1, 'one model without a score is not a listing without scores');
});

test('recentModels: newest first by the Hub\'s creation date; the same second keeps the listing\'s order; no date at all is an error', () => {
    const list = models();
    const recent = recentModels(list);
    assert.deepEqual(names(recent).slice(0, LIST_LENGTH), ['Kev-4B-LiteRT', 'Kev-0.8B-LiteRT', 'Qwen3-ASR-1.7B', 'GLiNER2.5-Multi-LiteRT', 'Julia-1-LiteRT',
        'decider-2b-vision-LiteRT', 'Audio8-TTS-Preview-0.6b', 'Fun-ASR-Nano-2512', 'GLiNER2.5-Decide-LiteRT', 'Nemotron-3-Diarization-LiteRT',
        'Laya-English-LiteRT', 'Laya-Multilingual-LiteRT']);
    assert.equal(recent.length, LIST_LENGTH + MORE_LENGTH, 'the first rows and the folded ones');
    assert.ok(recent.every((m, i) => i === 0 || m.created <= recent[i - 1].created));
    assert.deepEqual(names(list).slice(0, 2), ['gemma-4-E2B-it-litert-lm', 'Gemma3-1B-IT'], 'the input keeps its order');
    const at = (name, created) => ({ name, created });
    assert.deepEqual(names(recentModels([at('a', '2026-01-01T00:00:00.000Z'), at('b', '2026-02-01T00:00:00.000Z'), at('c', '2026-02-01T00:00:00.000Z'), at('d', '')])), ['b', 'c', 'a']);
    assert.equal(recentModels(list, 3).length, 3);
    assert.throws(() => recentModels(list.map(m => ({ ...m, created: '' }))), /carries no creation date/);
});

test('parsePicks reads the list under the heading with the word Picks, each item as written, with the org repos it links', () => {
    const picks = parsePicks(read('card.md'));
    assert.equal(picks.heading, '🌟 Community\'s Picks of the Week');
    assert.equal(picks.items.length, 6);
    assert.deepEqual([picks.dropped, picks.unlinked], [0, 0]);
    assert.deepEqual(picks.items[0].repos, ['litert-community/laya-LiteRT', 'litert-community/Laya-English-LiteRT', 'litert-community/Laya-Multilingual-LiteRT']);
    assert.match(picks.items[0].md, /^\*\*Laya Series\*\* \(`System One`\): \[laya-LiteRT\]/);
    assert.match(picks.items[0].md, /\(multilingual\) Convai Innovations' text encoders .* in one forward pass\.$/, 'the indented second line belongs to the item');
    assert.equal(picks.items[1].md, '**[PaddleOCR-VL-1.6](https://huggingface.co/litert-community/PaddleOCR-VL-1.6)** (`Image-Text-to-Text`) High-accuracy vision-language model tailored for robust on-device document understanding and OCR tasks.');
    assert.deepEqual(picks.items.slice(1).map(i => i.repos), [['litert-community/PaddleOCR-VL-1.6'], ['litert-community/decider-2b-vision-LiteRT'],
        ['litert-community/Spark-X2.5-4B'], ['litert-community/Audio8-TTS-Preview-0.6b'], ['litert-community/Nemotron-3-Diarization-LiteRT']]);
    assert.ok(!mds(picks).some(md => /Community Contributions|Are we missing/.test(md)), 'the list ends at the next heading');
});

test('parsePicks: which heading is the picks', () => {
    const one = parsePicks('# Org\n\nWe picked these.\n* not yet\n\n### our pick ###\n- [A](https://huggingface.co/litert-community/A) one\r\n+ **B** two\n\n   * [C](https://huggingface.co/litert-community/C/)\n### Next\n* not a pick\n');
    assert.equal(one.heading, 'our pick', 'at any level and in any case');
    assert.deepEqual(mds(one), ['[A](https://huggingface.co/litert-community/A) one', '**B** two', '[C](https://huggingface.co/litert-community/C/)']);
    assert.deepEqual(one.items.map(i => i.repos), [['litert-community/A'], [], ['litert-community/C']]);
    assert.equal(parsePicks('## **Picks**\n* x\n').heading, 'Picks', 'marks of the heading are dropped');
    assert.deepEqual(mds(parsePicks('## Picks\n* first\n\n## More picks\n* second\n')), ['first'], 'of two such headings, the first');
    const verb = parsePicks('## How to pick a model\n* step one\n\n# Pick your platform\n* Android\n\n## Our picks\n* real\n');
    assert.deepEqual([verb.heading, mds(verb)], ['Our picks', ['real']], 'a heading with "picks" before one with "pick"');
    assert.deepEqual(mds(parsePicks('## How to pick a model\n* step one\n')), ['step one'], '"pick" when no heading has "picks"');
    const elsewhere = '---\ntitle: README\n# our picks, a comment of the front matter\n---\n\n```\n## Picks in a code block\n* no\n```\n\n## [Models](https://huggingface.co/picks)\n* no\n\n## Pick-up guide\n* no\n\n';
    assert.throws(() => parsePicks(elsewhere), /no picks heading/, 'not in the front matter, a code block, a link\'s address or another word');
    assert.deepEqual(mds(parsePicks(elsewhere + '## The picks\n* yes\n')), ['yes']);
    assert.equal(parsePicks('## Picks\n* [x](https://huggingface.co/LiteRT-Community/Some-Model) and [y](https://huggingface.co/litert-community/some-model)\n').items[0].repos.length, 1, 'one repo, however the links spell it');
    assert.throws(() => parsePicks('# Org\n\n## Models\n* a\n'), /no picks heading/);
    assert.throws(() => parsePicks('We picked these.\n* a\n'), /no picks heading/, 'the word outside a heading is not the section');
    assert.throws(() => parsePicks('## Pickspace\n* a\n'), /no picks heading/, 'the word as a part of another');
    assert.throws(() => parsePicks('<!DOCTYPE html><html><body>Sign in</body></html>'), /no picks heading/);
    assert.throws(() => parsePicks(''), /no picks heading/);
    for (const v of [null, undefined, 7, {}, []]) assert.throws(() => parsePicks(v), /not text/);
});

test('parsePicks: what is an item', () => {
    assert.deepEqual(mds(parsePicks('## Picks\n1. one\n2) two\n10. ten\n')), ['one', 'two', 'ten'], 'numbered items');
    const nested = parsePicks('## Picks\n* first\n### Older\n* second\n#### Oldest\n* third\n## Next\n* not a pick\n# Picks again\n* nor this\n');
    assert.deepEqual([mds(nested), nested.dropped], [['first', 'second', 'third'], 2], 'a heading under it does not end the list, and is not an item');
    assert.deepEqual(mds(parsePicks('### Picks\n* a\n# Other\n* b\n')), ['a'], 'a heading above it ends the list');
    const loose = parsePicks('## Picks\nA line before the list.\n* first\n  second line\nthird line, not indented\n\nA paragraph after a blank line.\n* fourth\n');
    assert.deepEqual([mds(loose), loose.dropped], [['first second line third line, not indented', 'fourth'], 2], 'text outside an item is counted, not shown');
    const hidden = parsePicks('## Picks\n* shown\n<!--\n* hidden last week\n* hidden too\n-->\n* shown again <!-- a note -->\n<!-- never closed\n* gone\n');
    assert.deepEqual([mds(hidden), hidden.dropped], [['shown', 'shown again'], 0], 'what a comment hides is not a pick');
    assert.deepEqual(mds(parsePicks('## Picks\n* a\n\n* * *\n\n* b\n- - -\n* c\n***\n* d\n___\n')), ['a', 'b', 'c', 'd'], 'a rule between items is not an item');
    assert.deepEqual(mds(parsePicks('## Picks\n* Series\n    * [A](https://huggingface.co/litert-community/A)\n\t- [B](https://huggingface.co/litert-community/B)\n')),
        ['Series [A](https://huggingface.co/litert-community/A) [B](https://huggingface.co/litert-community/B)'], 'items under an item join it, without their marks');
    const separator = String.fromCharCode(0x2028);
    assert.deepEqual(mds(parsePicks(`## Picks\n* one\n* two${separator}still two\n* three\r* four\n`)), ['one', 'two still two', 'three', 'four'], 'every kind of line end');
    const fenced = parsePicks('## Picks\n* a\n```\n* in a code block\n```\n* b\n');
    assert.deepEqual([mds(fenced), fenced.dropped], [['a', 'b'], 3], 'a code block is counted, not shown');
    assert.throws(() => parsePicks('## Picks\n\nNothing here yet.\n\n## Next\n* a\n'), /no items/);
    assert.throws(() => parsePicks('## Picks\n* <div></div>\n* <!-- only a note -->\n'), /no items/, 'an item with no words is not an item');
});

test('parsePicks: tags, entities, long items and links to other sites', () => {
    const tags = parsePicks('## Picks <span>now</span>\n* <b>Bold</b> and <a href="https://example.com/x">a link</a><br>next a < b > c\n* uses the `<start_of_turn>` template and `<bos>`\n* <p>One.</p><p>Two.</p>\n');
    assert.deepEqual([tags.heading, mds(tags)], ['Picks now', ['Bold and a link next a < b > c', 'uses the `<start_of_turn>` template and `<bos>`', 'One. Two.']], 'tags go, code stays');
    assert.deepEqual(mds(parsePicks('## Picks\n* R&amp;D &lt;b&gt; &quot;q&quot; &#39;s&apos; a&nbsp;b &amp;lt; `&amp;`\n')), ['R&D <b> "q" \'s\' a b &lt; `&amp;`'], 'entities become their characters, once, outside code');
    const auto = parsePicks('## Picks\n* New: <https://huggingface.co/litert-community/Foo> and <https://example.com/x>\n');
    assert.deepEqual([auto.items[0].repos, auto.unlinked], [['litert-community/Foo'], 1], 'an address in angle brackets is a link');
    const links = parsePicks('## Picks\n* [x](https://huggingface.co/someone/else) [y](https://evil.example/litert-community/z) [w](https://huggingface.co/litert-community/w/tree/main) [d](https://ai.google.dev/edge/litert) [m](mailto:a@b.c)\n');
    assert.deepEqual(links.items[0].repos, [], 'only a link to a repo page of the org counts as a repo');
    assert.equal(links.unlinked, 2, 'the links to another site and scheme');
    const many = parsePicks('## Picks\n' + Array.from({ length: 40 }, (_, i) => `* item ${i}`).join('\n'));
    assert.deepEqual([many.items.length, many.dropped], [12, 28], 'at most twelve items; the rest are counted');
    const long = parsePicks('## Picks\n* ' + 'word '.repeat(196) + '[Foo](https://huggingface.co/litert-community/Foo) tail\n* short\n');
    assert.ok(long.items[0].md.length <= 1000 && long.items[0].md.endsWith('word …'), 'a long item is cut at a space, without the link the cut went through');
    assert.deepEqual([long.items[0].repos, long.dropped], [[], 1], 'and counted');
    assert.ok(parsePicks('## Picks\n* ' + 'x'.repeat(5000)).items[0].md.length <= 1002, 'an item without a space is cut too');
    assert.ok(!/[\uD800-\uDBFF]$/.test(parsePicks('## Picks\n* ' + '😀'.repeat(600)).items[0].md.replace(/ …$/, '')), 'not through a character');
});

test('parsePicks takes a card of any size in linear time', () => {
    const started = Date.now();
    for (const card of ['# ' + '#'.repeat(80000) + 'x\n## Picks\n* a\n', '## Picks\n* a ' + '<!--'.repeat(40000) + '\n', '## Picks\n* a ' + '<a'.repeat(60000) + '\n',
        '## Picks\n* a ' + '**'.repeat(60000) + '\n', '## Picks\n' + '* a\n'.repeat(60000), '## Picks\n* a ' + '[a]('.repeat(50000) + '\n']) {
        assert.equal(mds(parsePicks(card))[0].slice(0, 1), 'a');
    }
    assert.ok(Date.now() - started < 2000, `took ${Date.now() - started} ms`);
});

test('safeLink: https links to the four sites, as the browser will read them; null for any other', () => {
    assert.equal(safeLink('https://huggingface.co/litert-community/A'), 'https://huggingface.co/litert-community/A');
    assert.equal(safeLink('HTTPS://HuggingFace.co/a'), 'https://huggingface.co/a');
    assert.equal(safeLink('https://github.com/google-ai-edge/litert-samples'), 'https://github.com/google-ai-edge/litert-samples');
    assert.equal(safeLink('https://google-ai-edge.github.io/litert-samples/'), 'https://google-ai-edge.github.io/litert-samples/');
    assert.equal(safeLink('https://ai.google.dev/edge/litert'), 'https://ai.google.dev/edge/litert');
    for (const href of ['http://huggingface.co/a', 'https://huggingface.co.evil.example/a', 'https://huggingface.co@evil.example/a', 'https://user:pw@huggingface.co/a',
        'https://huggingface.co:8443/a', 'https://github.com/someone-else/repo', 'https://github.com/google-ai-edge/../someone-else/repo', 'https://evil.example/huggingface.co/',
        'javascript:alert(1)', 'data:text/html,x', '//huggingface.co/a', '/litert-community/A', 'not a url', '', null, undefined, 7]) assert.equal(safeLink(href), null, String(href));
});

test('inlineHtml: bold, code and links to the allowed sites become markup; everything else stays text', () => {
    assert.equal(inlineHtml('**[A](https://huggingface.co/litert-community/A)** (`Text Generation`) An edge LLM.'),
        '<b><a href="https://huggingface.co/litert-community/A" target="_blank" rel="noopener">A</a></b> (<code>Text Generation</code>) An edge LLM.');
    assert.equal(inlineHtml('[s](https://github.com/google-ai-edge/litert-samples) [d](https://ai.google.dev/edge/litert) [p](https://google-ai-edge.github.io/litert-samples/)').match(/<a /g).length, 3);
    assert.equal(inlineHtml('[x](https://evil.example/a) [y](javascript:alert(1)) [z](http://huggingface.co/a) [w](https://huggingface.co.evil.example/a) [v](https://huggingface.co@evil.example/a)'),
        'x [y](javascript:alert(1)) z w v', 'a link to anywhere else is its text; one that is not a link stays as written');
    assert.equal(inlineHtml('<img src=x onerror=alert(1)> & "q" \'s\''), '&lt;img src=x onerror=alert(1)&gt; &amp; &quot;q&quot; &#39;s&#39;');
    assert.equal(inlineHtml('[<b>x</b>](https://huggingface.co/a"onmouseover="x)'), '<a href="https://huggingface.co/a%22onmouseover=%22x" target="_blank" rel="noopener">&lt;b&gt;x&lt;/b&gt;</a>');
    assert.equal(inlineHtml('`<script>` and `**not bold**` and `[not a link](https://huggingface.co/a)`'), '<code>&lt;script&gt;</code> and <code>**not bold**</code> and <code>[not a link](https://huggingface.co/a)</code>');
    assert.equal(inlineHtml('**a **b** c**'), '<b>a </b>b<b> c</b>');
    assert.equal(inlineHtml('a ** b'), 'a ** b', 'an unclosed mark is text');
    assert.equal(inlineHtml('[**bold** and `code`](https://huggingface.co/a)'), '<a href="https://huggingface.co/a" target="_blank" rel="noopener"><b>bold</b> and <code>code</code></a>');
    const marks = ['**', '`', '[', ']', '(', ')', 'https://huggingface.co/a', 'https://evil.example/', 'javascript:x', '<img onerror=x>', '"', '\'', ' ', 'word', '&'];
    for (let seed = 1; seed <= 4000; seed++) {
        let n = seed;
        const text = Array.from({ length: 12 }, () => marks[(n = (n * 1103515245 + 12345) % 2147483648) % marks.length]).join('');
        const html = inlineHtml(text);
        const rest = html.replace(/<a href="https:\/\/huggingface\.co\/[^"<>]*" target="_blank" rel="noopener">|<\/a>|<\/?b>|<\/?code>/g, '');
        assert.ok(!/[<>]/.test(rest), `markup other than the three marks from ${JSON.stringify(text)}: ${html}`);
        assert.ok(!/<a [^>]*>(?:(?!<\/a>).)*<a /.test(html), `a link inside a link from ${JSON.stringify(text)}`);
    }
    for (const v of [null, undefined]) assert.equal(inlineHtml(v), '');
    assert.equal(inlineHtml(7), '7');
});

test('plainText drops the marks and keeps the words; withoutHtml drops the tags and keeps code', () => {
    assert.equal(plainText('Chat with **Gemma** on the [NPU](https://x.example/a), `fast`'), 'Chat with Gemma on the NPU, fast');
    assert.equal(plainText('  two   spaces\tand a tab '), 'two spaces and a tab');
    assert.equal(plainText(null), '');
    assert.equal(withoutHtml('a<br>b <i>c</i> `<i>d</i>` &amp; <https://x.example/y>'), 'a b c `<i>d</i>` & [https://x.example/y](https://x.example/y)');
    assert.equal(withoutHtml(null), '');
});

test('parseSamples reads the table by its header names and lists the rows that name a model of the org, in the README\'s order', () => {
    const { samples, skipped } = parseSamples(read('samples_litert.md'), 'samples/litert/');
    assert.equal(skipped, 0);
    assert.deepEqual(samples.map(s => s.name), ['google/sample_app_tpu', 'image_generation', 'image_segmentation', 'phototalk_sample_app', 'qualcomm/gemma3/npu',
        'semantic_similarity', 'speech_recognition', 'text_to_speech', 'zero_shot_classification']);
    assert.deepEqual(samples[1], {
        name: 'image_generation', path: 'samples/litert/image_generation', url: TREE + 'samples/litert/image_generation',
        task: 'Text to image (Bonsai Image 4B)', platform: 'iOS, macOS', models: ['litert-community/Bonsai-Image-ternary-4B'],
    });
    assert.deepEqual(samples[3].models, ['litert-community/gemma-4-E2B-it-litert-lm', 'litert-community/Gemma3-1B-IT', 'litert-community/FastVLM-0.5B']);
    assert.equal(samples[6].models.length, 6, 'every repo of the cell, by its link');
    assert.equal(samples[2].platform, 'Android (Kotlin on CPU/GPU and NPU; C++), iOS');
    assert.equal(samples[8].task, 'Zero-shot text classification with questions defined at run time (choice, score, yes/no)');
    const lm = parseSamples(read('samples_litert_lm.md'), 'samples/litert_lm/');
    assert.deepEqual(lm.samples.map(s => [s.path, s.platform, s.models.length]), [['samples/litert_lm/reachy-voice-robot', 'Raspberry Pi (Python)', 3]]);
});

test('parseSamples: rows it cannot read are skipped and counted; every samples table of the README is read; one without the table throws', () => {
    const header = '| Sample | Task | Platform | Model |\n|---|---|---|---|\n';
    const org = name => `[${name}](${ORG}${name})`;
    const rows = parseSamples(header
        + `| [\`a/\`](a/) | A | Android | ${org('A')} |\n`
        + `| [\`b/\`](b/), [\`c/\`](c/) | two directories | Android | ${org('B')} |\n`
        + `| [\`../up/\`](../up/) | a path outside | Android | ${org('C')} |\n`
        + `| [\`d/\`](d/) | a \\| in a cell | Android | ${org('D')} |\n`
        + `| [\`e/\`](e/) | no model of the org | Android | [x](https://huggingface.co/qualcomm/MobileNet-v2), a file in the tree |\n`
        + `| [\`f/\`](./f/g) | F<br>two &amp; three | <b>iOS</b> | ${org('F')}, ${org('f')} and [n](https://evil.example/litert-community/N) |\n`
        + `| [\`a/\`](a/) | the same directory again | Android | ${org('A2')} |\n`
        + 'text after the table\n\n'
        + '| Demo | What it does |\n|---|---|\n| [z](z/) | not a samples table | \n\n'
        + header + `| [\`z/\`](z/) | a second table | Android | ${org('Z')} |\n`, 'samples/x/');
    assert.deepEqual(rows.samples.map(s => [s.name, s.path, s.models]), [['a', 'samples/x/a', ['litert-community/A', 'litert-community/A2']],
        ['f/g', 'samples/x/f/g', ['litert-community/F']], ['z', 'samples/x/z', ['litert-community/Z']]], 'a directory once, with the models of its rows');
    assert.equal(rows.skipped, 3, 'two directories, a path outside the directory, a row with another cell count');
    assert.deepEqual([rows.samples[1].task, rows.samples[1].platform], ['F two & three', 'iOS'], 'cells without their tags');
    const fewer = parseSamples(`| MODEL | Sample |\n|---|---|\n| ${org('A')} | [a](a) |\n`, 'samples/x/');
    assert.deepEqual(fewer.samples, [{ name: 'a', path: 'samples/x/a', url: TREE + 'samples/x/a', task: '', platform: '', models: ['litert-community/A'] }], 'the columns are found by name, in any order and case');
    assert.deepEqual(parseSamples(header, 'samples/x/'), { samples: [], skipped: 0 }, 'an empty table is not a failure');
    const hostile = parseSamples(header + `| [<img src=x onerror=x>](javascript:alert(1)) | x | x | ${org('A')} |\n| [ok](ok/) | a < b | 1 > 0 | ${org('A')} |\n`, 'samples/x/');
    assert.equal(hostile.skipped, 1);
    assert.deepEqual([hostile.samples[0].url, hostile.samples[0].task, hostile.samples[0].platform], [TREE + 'samples/x/ok', 'a < b', '1 > 0'], 'the cells are text for the page to escape');
    assert.throws(() => parseSamples('# Samples\n| Demo | What it does |\n|---|---|\n| a | b |\n', 'samples/x/'), /no samples table/);
    assert.throws(() => parseSamples('', 'samples/x/'), /no samples table/);
    for (const v of [null, undefined, 7, {}]) assert.throws(() => parseSamples(v, 'samples/x/'), /not text/);
});

test('samplesByModel gives each repo its samples; samplesFor keeps the samples of the repos the page lists', () => {
    const { samples } = parseSamples(read('samples_litert.md'), 'samples/litert/');
    const byModel = samplesByModel(samples);
    assert.deepEqual(byModel.get('litert-community/gemma-4-e2b-it-litert-lm').map(s => s.name), ['google/sample_app_tpu', 'phototalk_sample_app'], 'by the repo id in lower case');
    assert.deepEqual(byModel.get('litert-community/laya-multilingual-litert').map(s => s.name), ['zero_shot_classification']);
    assert.equal(byModel.get('litert-community/kev-4b-litert'), undefined);
    assert.equal(byModel.size, 14, 'one entry for each repo the rows name');
    assert.deepEqual([...samplesByModel([]).keys()], []);
    const list = models();
    const listed = new Set([...trendingModels(list), ...recentModels(list)].map(m => m.id.toLowerCase()));
    assert.deepEqual(samplesFor(samples, listed).map(s => s.name), ['google/sample_app_tpu', 'image_generation', 'phototalk_sample_app', 'qualcomm/gemma3/npu',
        'semantic_similarity', 'speech_recognition', 'text_to_speech', 'zero_shot_classification'], 'in the order of the table; a sample of a repo on neither list is left out');
    assert.deepEqual(samplesFor(samples, new Set(['litert-community/matcha-tts'])).map(s => s.name), ['text_to_speech']);
    assert.deepEqual(samplesFor(samples, new Set()), [], 'no repo listed, no sample');
});

test('parseDemos lists the Spaces of the collection in its order', () => {
    const demos = parseDemos(fixture('demos.json'));
    assert.deepEqual(demos, [
        { id: 'tylermullen/Gemma4', title: 'Gemma4', description: 'Chat with Gemma 4 models fully in the browser via MediaPipe', likes: 33 },
        { id: 'tylermullen/Gemma3n', title: 'Gemma3n', description: 'Gemma 3n (📷+🎤+💬) fully in the browser w/ MediaPipe LLM', likes: 10 },
        { id: 'tylermullen/Gemma3', title: 'Gemma3', description: 'Chat webapp running Gemma 3 models fully in the browser', likes: 16 },
    ]);
    const odd = parseDemos({ items: [null, 7, { type: 'model', id: 'a/b' }, { type: 'space', id: '<img src=x>' }, { type: 'space', id: 'a/b', private: true },
        { type: 'space', id: 'ok/space', title: 7, shortDescription: null, likes: 'many' }, { type: 'space', id: 'ok/two', title: '  ', likes: -1 }] });
    assert.deepEqual(odd, [{ id: 'ok/space', title: 'space', description: '', likes: 0 }, { id: 'ok/two', title: 'two', description: '', likes: 0 }]);
    assert.equal(parseDemos({ items: Array.from({ length: 40 }, (_, i) => ({ type: 'space', id: `a/s${i}` })) }).length, 12);
    assert.deepEqual(parseDemos({ items: [] }), []);
    for (const v of [null, undefined, 7, 'x', [], {}, { items: 'none' }, { error: 'Not found' }]) assert.throws(() => parseDemos(v), /not a collection/);
});

test('nextLink reads the next page out of a Link header in the forms the header allows', () => {
    assert.equal(nextLink('<https://huggingface.co/api/models?cursor=abc>; rel="next"'), 'https://huggingface.co/api/models?cursor=abc');
    assert.equal(nextLink('<https://x/prev>; rel="prev", <https://x/next>; rel="next"'), 'https://x/next');
    assert.equal(nextLink('<https://x/next>; title="a, b"; rel=next'), 'https://x/next', 'another parameter first, rel without quotes');
    assert.equal(nextLink('<https://x/both>; rel="prev next"'), 'https://x/both', 'two relations in one');
    assert.equal(nextLink('<https://x/a,b?c=1,2>; rel="next"'), 'https://x/a,b?c=1,2', 'a comma in the address');
    assert.equal(nextLink('<https://x/first>; REL="NEXT"'), null, 'the relation is spelled in lower case');
    for (const header of [null, '', '<https://x/prev>; rel="prev"', '<https://x/n>; rel="nextish"', 'rel="next"']) assert.equal(nextLink(header), null);
});

test('loadModels follows the Hub paging header to its end; a listing that never ends is an error', async () => {
    const all = fixture('models.json');
    const calls = [];
    const get = async url => {
        calls.push(url);
        const page = calls.length - 1;
        return { data: all.slice(page * 12, page * 12 + 12), link: (page + 1) * 12 < all.length ? `<https://hub.test/page${page + 1}>; rel="next"` : null };
    };
    const paged = await loadModels(get);
    assert.deepEqual(calls, [MODELS_URL, 'https://hub.test/page1', 'https://hub.test/page2', 'https://hub.test/page3']);
    assert.deepEqual(names(paged), names(models()), 'the pages keep the order of the listing');
    await assert.rejects(loadModels(async () => ({ data: { error: 'Rate limit reached' }, link: null })), /not a model list/);
    await assert.rejects(loadModels(async () => ({ data: [], link: null })), /came back empty/);
    await assert.rejects(loadModels(async () => { throw new Error('HTTP 429'); }), /HTTP 429/);
    await assert.rejects(loadModels(async url => { if (url !== MODELS_URL) throw new Error('HTTP 500'); return { data: all.slice(0, 12), link: '<https://hub.test/page1>; rel="next"' }; }), /HTTP 500/, 'a second page that fails fails the list');
    let pages = 0;
    await assert.rejects(loadModels(async () => ({ data: [all[0]], link: `<https://hub.test/again${++pages}>; rel="next"` })), /did not end/);
    assert.equal(pages, 20);
});

test('loadSamples: one table that cannot be read leaves the other; none is an error', async () => {
    const texts = { [SAMPLE_INDEXES[0].url]: read('samples_litert.md'), [SAMPLE_INDEXES[1].url]: read('samples_litert_lm.md') };
    const from = map => async url => { if (map[url] instanceof Error) throw map[url]; return { body: map[url] }; };
    const both = await loadSamples(from(texts));
    assert.deepEqual([both.samples.length, both.failed], [10, []]);
    assert.equal(both.samples.at(-1).path, 'samples/litert_lm/reachy-voice-robot', 'the tables in the order the page names them');
    assert.equal(both.byModel.get('litert-community/gemma-4-e2b-it-litert-lm').length, 3);
    const one = await loadSamples(from({ ...texts, [SAMPLE_INDEXES[0].url]: new Error('HTTP 404') }));
    assert.deepEqual([one.samples.length, one.failed], [1, [{ dir: 'samples/litert/', why: 'HTTP 404' }]]);
    const shape = await loadSamples(from({ ...texts, [SAMPLE_INDEXES[1].url]: '# no table\n' }));
    assert.deepEqual([shape.samples.length, shape.failed], [9, [{ dir: 'samples/litert_lm/', why: 'no samples table in the README' }]]);
    await assert.rejects(loadSamples(from({ [SAMPLE_INDEXES[0].url]: new Error('HTTP 404'), [SAMPLE_INDEXES[1].url]: new Error('no answer in 8 s') })), /HTTP 404/);
});

test('esc and the count format', () => {
    assert.equal(esc('<img src=x onerror="a(\'b\')">&'), '&lt;img src=x onerror=&quot;a(&#39;b&#39;)&quot;&gt;&amp;');
    assert.equal(esc(null), '');
    assert.equal(fmtCount(1398895), '1,398,895');
    assert.equal(fmtCount(0.5), '0.5');
    for (const v of [null, undefined, 'abc', NaN, Infinity]) assert.equal(fmtCount(v), '');
});

test('the page files: the CSP reaches the five source URLs and nothing else; one name and one draft state in the page and its card; every id the script looks up', () => {
    const html = file('../space/index.html');
    const card = file('../space/README.md');
    const script = file('../space/app.js');
    const connect = html.match(/connect-src ([^;"]+)/)[1].trim().split(/\s+/);
    assert.deepEqual([...connect].sort(), ['\'self\'', MODELS_URL.split('?')[0], CARD_URL, ...SAMPLE_INDEXES.map(s => s.url), DEMOS_URL].sort());
    assert.equal(new Set(connect).size, 6);
    const title = html.match(/<title>([^<]*)<\/title>/)[1];
    const brand = html.match(/<span class="brand">([^<]*)<\/span>/)[1];
    const heading = html.match(/<h1>([^<]*)<\/h1>/)[1];
    const cardTitle = card.match(/^title: (.*)$/m)[1];
    const name = title.replace(' (draft)', '');
    assert.deepEqual([brand, cardTitle.replace(' (draft)', '')], [name, name], 'the name of the page');
    assert.ok(heading.startsWith(name), 'the first heading starts with the name');
    const drafts = [title.endsWith(' (draft)'), /<p class="draft" role="note">Draft · for review<\/p>/.test(html), cardTitle.endsWith(' (draft)'),
        /^short_description: '?Draft: /m.test(card), /^Draft, for review\. /m.test(card)];
    assert.equal(new Set(drafts).size, 1, `the draft marks are all there or all gone: ${drafts.join(' ')}`);
    const versions = [...html.matchAll(/(?:style\.css|app\.js)\?v=(\d+)/g)].map(m => m[1]);
    assert.deepEqual([versions.length, new Set(versions).size], [2, 1], 'the stylesheet and the script carry the same version');
    assert.ok(card.match(/^short_description: '?(.*?)'?$/m)[1].length <= 60, 'the Hub takes a short description of 60 characters at most');
    assert.ok(!/<[a-z]+[^>]* style=|<script(?![^>]* src=)|\son[a-z]+=/.test(html), 'no inline style or script: the CSP blocks them');
    const ids = [...html.matchAll(/ id="([^"]+)"/g)].map(m => m[1]);
    assert.equal(new Set(ids).size, ids.length, 'an id that is on the page twice');
    const parts = ['trending', 'recent', 'picks', 'samples', 'demos'];
    const looked = [...parts.flatMap(p => [p, `${p}Loading`, `${p}Error`, `${p}ErrorWhy`, `${p}List`]), ...['trending', 'recent'].flatMap(p => [`${p}More`, `${p}MoreList`, `${p}MoreCount`]),
        ...[...script.matchAll(/\$\('([A-Za-z]+)'\)/g)].map(m => m[1])];
    assert.deepEqual(looked.filter(id => !ids.includes(id)), [], 'an id the script looks up is not on the page');
    assert.ok(looked.includes('picksTitle') && looked.includes('samplesPartialWhich'), 'the ids the script names are found in it');
    for (const p of parts) assert.match(html, new RegExp(`\\(<span id="${p}ErrorWhy"></span>\\)`), `${p}: the reason sits in the notice`);
    assert.ok(html.includes(`shows its first ${LIST_LENGTH} models and folds up to ${MORE_LENGTH} more`), 'the lengths the page names are the lengths of its lists');
    assert.ok(html.includes('shows up to 12 picks, each cut at 1,000 characters, and up to 12 demos') && /MAX_PICKS = 12;\nconst MAX_PICK_CHARS = 1000;\nconst MAX_DEMOS = 12;/.test(script), 'the limits the page names are the script\'s');
});

// Trending models: the org's models in the order of the Hub's Trending sort and by creation date, the picks
// on the litert-community organization card, the samples of litert-samples that run them, and the org's
// Demos collection. All four sources are read when the page loads, and a source that cannot be read leaves
// the other parts of the page as they are.

const HUB = "https://huggingface.co";
const HUB_MODELS_API = "https://huggingface.co/api/models";
const FIELDS = ['trendingScore', 'likes', 'downloads', 'createdAt', 'pipeline_tag', 'gated', 'private', 'siblings', 'cardData'];
export const MODELS_URL = HUB_MODELS_API + '?author=litert-community&sort=trendingScore&direction=-1&limit=1000' + FIELDS.map(f => `&expand=${f}`).join('');
export const CARD_URL = "https://huggingface.co/spaces/litert-community/README/raw/main/README.md";
// The sample tables the page reads, each with the directory its rows are relative to.
export const SAMPLE_INDEXES = [
    { url: "https://google-ai-edge.github.io/litert-samples/samples/litert/README.md", dir: 'samples/litert/' },
    { url: "https://google-ai-edge.github.io/litert-samples/samples/litert_lm/README.md", dir: 'samples/litert_lm/' },
];
export const DEMOS_URL = "https://huggingface.co/api/collections/litert-community/demos-68efdb213c93efee10c2116f";
const TREE = "https://github.com/google-ai-edge/litert-samples/tree/main/";
const TIMEOUT_MS = { models: 20000, card: 8000, samples: 8000, demos: 8000 };
const MAX_PAGES = 20;
// A model list shows its first rows and keeps the next ones behind a fold.
export const LIST_LENGTH = 12;
export const MORE_LENGTH = 24;
const MAX_CARD_CHARS = 200000;
const MAX_PICKS = 12;
const MAX_PICK_CHARS = 1000;
const MAX_DEMOS = 12;

export const FORMATS = ['tflite', 'litertlm', 'task'];
// Where a link of the organization card may lead: a host and the start of the path. A link to anywhere
// else is shown as its text.
const LINK_SITES = [['huggingface.co', '/'], ['github.com', '/google-ai-edge/'], ['google-ai-edge.github.io', '/'], ['ai.google.dev', '/']];

// ---- formatting -------------------------------------------------------------------------------

export function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
export const fmtCount = n => (Number.isFinite(n) ? n.toLocaleString('en-US') : '');
const day = iso => (typeof iso === 'string' ? iso.slice(0, 10) : '');
const plural = (n, one) => `${fmtCount(n)} ${one}${n === 1 ? '' : 's'}`;
const cmp = (x, y) => (x < y ? -1 : x > y ? 1 : 0);
const repoPath = id => String(id).split('/').map(encodeURIComponent).join('/');
const repoUrl = (id, kind = '') => esc(`${HUB}/${kind}${repoPath(id)}`);
const REPO_ID = /^[A-Za-z0-9][\w.-]*\/(?=[\w.-]*\w)[\w.-]+$/;
const ORG_REPO_URL = /^https:\/\/huggingface\.co\/(litert-community\/[\w.-]+)\/?$/i;
const SAMPLE_PATH = /^([A-Za-z0-9_-][A-Za-z0-9._-]*\/)*[A-Za-z0-9_-][A-Za-z0-9._-]*\/?$/;
const MD_LINK = /\[([^\]]*)\]\(([^()\s]+)\)/g;
// A repo id as a key: the Hub resolves an id in any case.
const key = id => String(id).toLowerCase();
// The repos of the org a line of Markdown links to, each once, as the line spells them.
function orgRepos(md) {
    const repos = new Map();
    for (const m of md.matchAll(MD_LINK)) {
        const id = (m[2].match(ORG_REPO_URL) || [])[1];
        if (id && !repos.has(key(id))) repos.set(key(id), id);
    }
    return [...repos.values()];
}

// The address a link may carry: https, one of the sites above, no user name or port. Null for any other.
export function safeLink(href) {
    let url;
    try {
        url = new URL(href);
    } catch (err) {
        return null;
    }
    const known = LINK_SITES.some(([host, path]) => url.hostname === host && url.pathname.startsWith(path));
    return url.protocol === 'https:' && known && !url.username && !url.password && !url.port ? url.href : null;
}

// The bold, code and link marks of a line of Markdown as HTML; everything else as text.
const INLINE = /\*\*(.+?)\*\*|`([^`]+)`|\[([^\]]*)\]\(([^()\s]+)\)/g;
export function inlineHtml(md) {
    const text = String(md == null ? '' : md);
    let out = '';
    let at = 0;
    for (const m of text.matchAll(INLINE)) {
        out += esc(text.slice(at, m.index));
        at = m.index + m[0].length;
        const href = m[4] == null ? null : safeLink(m[4]);
        if (m[1] != null) out += `<b>${inlineHtml(m[1])}</b>`;
        else if (m[2] != null) out += `<code>${esc(m[2])}</code>`;
        else if (href) out += `<a href="${esc(href)}" target="_blank" rel="noopener">${inlineHtml(m[3])}</a>`;
        else out += inlineHtml(m[3]);
    }
    return out + esc(text.slice(at));
}
// The same line without its marks.
export function plainText(md) {
    return String(md == null ? '' : md).replace(MD_LINK, '$1').replace(/\*\*|`/g, '').replace(/\s+/g, ' ').trim();
}

// A line of Markdown without its HTML: an autolink becomes a link, a tag that breaks a line becomes a
// space, any other tag is dropped, and the named entities become their characters. A code span is kept as
// it is.
const BREAKING_TAG = /<\/?(?:br|p|div|li|ul|ol|tr|td|th|table|h[1-6])\b[^<>]*>/gi;
const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", '#39': "'", nbsp: ' ' };
export function withoutHtml(md) {
    const plain = part => part
        .replace(/<(https?:\/\/[^\s<>]+)>/g, '[$1]($1)')
        .replace(BREAKING_TAG, ' ')
        .replace(/<\/?[A-Za-z][^<>]*>/g, '')
        .replace(/&(amp|lt|gt|quot|apos|#39|nbsp);/g, (entity, name) => ENTITIES[name]);
    return String(md == null ? '' : md).split(/(`[^`]*`)/).map((part, i) => (i % 2 ? part : plain(part))).join('').replace(/\s+/g, ' ').trim();
}

// The text without its HTML comments; a comment that is never closed runs to the end.
function withoutComments(text) {
    let out = '';
    let at = 0;
    for (;;) {
        const open = text.indexOf('<!--', at);
        if (open < 0) return out + text.slice(at);
        out += text.slice(at, open);
        const close = text.indexOf('-->', open + 4);
        if (close < 0) return out;
        at = close + 3;
    }
}

// ---- data -------------------------------------------------------------------------------------

// The checkpoints a card's base_model field names, kept only as repo ids.
export function sourceModels(card) {
    const v = card && typeof card === 'object' ? card.base_model : null;
    const list = Array.isArray(v) ? v : [v];
    return [...new Set(list.filter(s => typeof s === 'string' && REPO_ID.test(s)))].slice(0, 8);
}

// The URL of a `Link` header's rel="next" entry, the one the Hub sends when a listing has another page.
export function nextLink(header) {
    if (!header) return null;
    for (const m of header.matchAll(/<([^>]*)>((?:[^,<"]|"[^"]*")*)/g)) {
        const rel = m[2].match(/;\s*rel\s*=\s*(?:"([^"]*)"|([^;\s]*))/i);
        if (rel && (rel[1] || rel[2] || '').split(/\s+/).includes('next')) return m[1];
    }
    return null;
}

// Public repos that hold at least one file besides the card and .gitattributes, in the order the Hub sent
// them. A count the Hub did not send is zero, which the page does not show; a trending score it did not
// send is null.
export function normalizeModels(list) {
    if (!Array.isArray(list)) throw new Error('the answer is not a model list');
    const count = v => (Number.isFinite(v) && v > 0 ? v : 0);
    return list.filter(m => m && typeof m.id === 'string' && REPO_ID.test(m.id) && !m.private).map(m => {
        const names = Array.isArray(m.siblings) ? m.siblings.map(s => s && s.rfilename).filter(n => typeof n === 'string') : null;
        const exts = new Set((names || []).filter(n => n.includes('.')).map(n => n.slice(n.lastIndexOf('.') + 1).toLowerCase()));
        return {
            id: m.id,
            name: m.id.slice(m.id.indexOf('/') + 1),
            task: typeof m.pipeline_tag === 'string' && m.pipeline_tag ? m.pipeline_tag : null,
            likes: count(m.likes),
            downloads: count(m.downloads),
            trending: Number.isFinite(m.trendingScore) && m.trendingScore >= 0 ? m.trendingScore : null,
            created: typeof m.createdAt === 'string' ? m.createdAt : '',
            gated: Boolean(m.gated),
            empty: names !== null && names.every(n => n === 'README.md' || n === '.gitattributes'),
            formats: FORMATS.filter(ext => exts.has(ext)),
            source: sourceModels(m.cardData),
        };
    }).filter(m => !m.empty);
}

// The Hub's Trending order: the models whose trending score is above zero, in the order the Hub sent them.
// Throws when no model carries a score, so a listing without the field is not shown as "none trending".
export function trendingModels(models, limit = LIST_LENGTH + MORE_LENGTH) {
    if (models.length && models.every(m => m.trending === null)) throw new Error('the model list carries no trending score');
    return models.filter(m => m.trending > 0).slice(0, limit);
}

// The models the Hub created last, newest first; two created in the same second keep the order of the
// list. Throws when no model carries a creation date.
export function recentModels(models, limit = LIST_LENGTH + MORE_LENGTH) {
    if (models.length && models.every(m => !m.created)) throw new Error('the model list carries no creation date');
    return models.filter(m => m.created).sort((a, b) => cmp(b.created, a.created)).slice(0, limit);
}

// A heading line's level and title; null for any other line.
function headingOf(line) {
    const m = line.match(/^ {0,3}(#{1,6})[ \t]+(.*)$/);
    if (!m) return null;
    let end = m[2].length;
    while (end > 0 && ' \t#'.includes(m[2][end - 1])) end--;
    return { level: m[1].length, title: m[2].slice(0, end) };
}

// The lines of a card after its front matter, without HTML comments, each marked when it is inside a fenced
// code block.
function cardLines(text) {
    const lines = withoutComments(text.slice(0, MAX_CARD_CHARS).replace(/[\u2028\u2029]/g, ' ')).split(/\r\n?|\n/);
    const frontMatterEnd = lines[0].trim() === '---' ? lines.findIndex((line, i) => i > 0 && line.trim() === '---') : -1;
    let fence = null;
    return lines.slice(frontMatterEnd + 1).map(line => {
        const mark = (line.match(/^ {0,3}(`{3,}|~{3,})/) || [])[1];
        if (fence) {
            if (mark && mark[0] === fence[0] && mark.length >= fence.length) fence = null;
            return { line, code: true };
        }
        if (mark) fence = mark;
        return { line, code: Boolean(mark) };
    });
}

// An item no longer than the page shows: a longer one is cut at a space, without the link the cut went
// through.
function cutItem(md) {
    if (md.length <= MAX_PICK_CHARS) return { md, cut: false };
    const space = md.lastIndexOf(' ', MAX_PICK_CHARS);
    const short = md.slice(0, space > 0 ? space : MAX_PICK_CHARS).replace(/\[[^[\]]*(?:\]\([^()]*)?$/, '').replace(/[\uD800-\uDBFF]$/, '');
    return { md: short.trimEnd() + ' …', cut: true };
}

// The picks on the organization card: the list under the first heading that has the word "picks" (or
// "pick", when no heading has "picks"), up to the next heading of that level or above. Each item is kept as
// it is written, without HTML, with the repos of the org it links to. `dropped` counts what of the section
// the page does not show (a line that is in no item, an item past the twelfth, an item cut short) and
// `unlinked` the links it shows as text. Throws when the card has no such list.
export function parsePicks(text) {
    if (typeof text !== 'string') throw new Error('the answer is not text');
    const lines = cardLines(text);
    const headings = [];
    lines.forEach(({ line, code }, at) => {
        const h = code ? null : headingOf(line);
        if (h) headings.push({ at, level: h.level, title: plainText(withoutHtml(h.title)) });
    });
    const named = word => headings.find(h => word.test(h.title));
    const start = named(/\bpicks\b(?!-)/i) || named(/\bpick\b(?!-)/i);
    if (!start) throw new Error('the organization card has no picks heading');
    const found = [];
    let open = false;
    let dropped = 0;
    for (const { line, code } of lines.slice(start.at + 1)) {
        const h = code ? null : headingOf(line);
        if (h && h.level <= start.level) break;
        const plain = !h && !code;
        const bullet = plain ? line.match(/^ {0,3}(?:[*+-]|\d{1,3}[.)])[ \t]+(.*)$/) : null;
        if (plain && /^ {0,3}([*_-])(?:[ \t]*\1){2,}[ \t]*$/.test(line)) open = false;
        else if (bullet) { found.push(bullet[1]); open = true; }
        else if (!line.trim()) open = false;
        else if (plain && open) found[found.length - 1] += ' ' + line.replace(/^\s+(?:[*+-]|\d{1,3}[.)])[ \t]+/, '');
        else { dropped++; open = false; }
    }
    const items = found.map(withoutHtml).filter(Boolean);
    const kept = items.slice(0, MAX_PICKS).map(cutItem);
    if (!kept.length) throw new Error('the picks on the organization card have no items');
    return {
        heading: start.title,
        items: kept.map(({ md }) => ({ md, repos: orgRepos(md) })),
        dropped: dropped + items.length - kept.length + kept.filter(item => item.cut).length,
        unlinked: kept.reduce((n, { md }) => n + [...md.matchAll(MD_LINK)].filter(m => !safeLink(m[2])).length, 0),
    };
}

// The "Samples" tables of a README in litert-samples: the rows whose Model cell links a repo of the org,
// each with its directory, task, platform and those repos. A table is one whose header names a Sample and
// a Model column; the columns are found by those names. A row the page cannot read as one directory (two
// links, a path outside the directory, a cell count that differs from the header's) is skipped and counted
// when it names a repo of the org.
export function parseSamples(text, dir) {
    if (typeof text !== 'string') throw new Error('the answer is not text');
    const cells = line => line.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
    const words = md => plainText(withoutHtml(md));
    const samples = new Map();
    let cols = null;
    let tables = 0;
    let skipped = 0;
    for (const line of text.split(/\r\n?|\n/)) {
        if (!line.trim().startsWith('|')) { cols = null; continue; }
        const row = cells(line);
        if (cols === null) {
            const names = row.map(c => c.toLowerCase());
            const samplesTable = names.includes('sample') && names.includes('model');
            cols = samplesTable && { sample: names.indexOf('sample'), model: names.indexOf('model'), task: names.indexOf('task'), platform: names.indexOf('platform'), width: row.length };
            tables += Number(samplesTable);
            continue;
        }
        if (!cols || row.every(c => /^:?-+:?$/.test(c))) continue;
        if (row.length !== cols.width) { if (orgRepos(line).length) skipped++; continue; }
        const models = orgRepos(row[cols.model]);
        if (!models.length) continue;
        const paths = [...row[cols.sample].matchAll(MD_LINK)].map(m => m[2].replace(/^\.\//, '')).filter(p => SAMPLE_PATH.test(p));
        if (paths.length !== 1) { skipped++; continue; }
        const name = paths[0].replace(/\/$/, '');
        const known = samples.get(name);
        if (known) {
            known.models = [...new Map([...known.models, ...models].map(id => [key(id), id])).values()];
            continue;
        }
        samples.set(name, {
            name,
            path: dir + name,
            url: TREE + dir + name,
            task: cols.task < 0 ? '' : words(row[cols.task]),
            platform: cols.platform < 0 ? '' : words(row[cols.platform]),
            models,
        });
    }
    if (!tables) throw new Error('no samples table in the README');
    return { samples: [...samples.values()], skipped };
}

// The samples that run each repo, by the repo's key.
export function samplesByModel(samples) {
    const byModel = new Map();
    for (const s of samples) for (const id of s.models) byModel.set(key(id), [...(byModel.get(key(id)) || []), s]);
    return byModel;
}

// The samples that run at least one of the repos with the given keys, in the order of the tables.
export function samplesFor(samples, keys) {
    return samples.filter(s => s.models.some(id => keys.has(key(id))));
}

// The Spaces of a collection, in its order. Throws when the answer is not a collection.
export function parseDemos(data) {
    if (!data || typeof data !== 'object' || !Array.isArray(data.items)) throw new Error('the answer is not a collection');
    return data.items.filter(i => i && i.type === 'space' && typeof i.id === 'string' && REPO_ID.test(i.id) && !i.private).slice(0, MAX_DEMOS).map(i => ({
        id: i.id,
        title: typeof i.title === 'string' && i.title.trim() ? i.title.trim() : i.id.slice(i.id.indexOf('/') + 1),
        description: typeof i.shortDescription === 'string' ? i.shortDescription.trim() : '',
        likes: Number.isFinite(i.likes) && i.likes > 0 ? i.likes : 0,
    }));
}

// ---- loading ----------------------------------------------------------------------------------

// The reason a request failed, in the page's own words: a browser's wording of a lost connection differs
// from one browser to the next.
async function getText(url, timeoutMs, accept) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    try {
        const res = await fetch(url, { signal: ctl.signal, headers: { Accept: accept } });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return { body: await res.text(), link: res.headers.get('Link') };
    } catch (err) {
        if (err.name === 'AbortError') throw new Error(`no answer in ${timeoutMs / 1000} s`);
        throw new Error(err instanceof TypeError ? 'no connection' : err.message || String(err));
    } finally {
        clearTimeout(timer);
    }
}

async function getJson(url, timeoutMs = TIMEOUT_MS.models) {
    const { body, link } = await getText(url, timeoutMs, 'application/json');
    try {
        return { data: JSON.parse(body), link };
    } catch (err) {
        throw new Error('the answer is not JSON');
    }
}

// Every page of the listing, or an error: a part of the list is never shown as the whole.
export async function loadModels(get = getJson) {
    const all = [];
    let url = MODELS_URL;
    for (let page = 0; url; page++) {
        if (page === MAX_PAGES) throw new Error('the listing did not end');
        const { data, link } = await get(url);
        if (!Array.isArray(data)) throw new Error('the answer is not a model list');
        all.push(...data);
        url = nextLink(link);
    }
    const models = normalizeModels(all);
    if (!models.length) throw new Error('the list came back empty');
    return models;
}

async function loadPicks() {
    const picks = parsePicks((await getText(CARD_URL, TIMEOUT_MS.card, 'text/plain, text/markdown')).body);
    if (picks.dropped) console.warn(`${picks.dropped} line(s) or item(s) of the picks are left out or cut short`);
    if (picks.unlinked) console.warn(`${picks.unlinked} link(s) of the picks lead outside the sites this page links to and are shown as text`);
    return { ...picks, keys: new Set(picks.items.flatMap(i => i.repos).map(key)) };
}

// The samples of every table that could be read, and the directories of those that could not. Throws when
// none could.
export async function loadSamples(get = url => getText(url, TIMEOUT_MS.samples, 'text/markdown, text/plain')) {
    const answers = await Promise.allSettled(SAMPLE_INDEXES.map(async ({ url, dir }) => parseSamples((await get(url)).body, dir)));
    const samples = [];
    const failed = [];
    answers.forEach((a, i) => {
        const { dir } = SAMPLE_INDEXES[i];
        if (a.status === 'rejected') {
            console.warn(`${dir}README.md could not be read: ${a.reason.message}`);
            return void failed.push({ dir, why: a.reason.message });
        }
        samples.push(...a.value.samples);
        if (a.value.skipped) console.warn(`${a.value.skipped} row(s) of ${dir}README.md could not be read as one sample directory and are left out`);
    });
    if (failed.length === SAMPLE_INDEXES.length) throw new Error(failed[0].why);
    return { samples, failed, byModel: samplesByModel(samples) };
}

async function loadDemos() {
    const demos = parseDemos((await getJson(DEMOS_URL, TIMEOUT_MS.demos)).data);
    if (!demos.length) throw new Error('the collection lists no Space');
    return demos;
}

// ---- page -------------------------------------------------------------------------------------

const $ = id => document.getElementById(id);
// What each source gave: null while it is being read, then its value or the reason it failed.
const got = { models: null, picks: null, samples: null, demos: null };
const value = name => (got[name] && got[name].ok ? got[name].value : null);
const logged = new Set();

function samplesOf(id) {
    const s = value('samples');
    return (s && s.byModel.get(key(id))) || [];
}

// The mark of a model that samples run: the sample's own directory for one, the samples part for several.
function sampleTags(ids) {
    const samples = [...new Set(ids.flatMap(samplesOf))];
    if (samples.length > 1) return `<a class="tag sample" href="#samples">${esc(plural(samples.length, 'sample'))}</a>`;
    return samples.map(s => `<a class="tag sample" href="${esc(s.url)}" target="_blank" rel="noopener">Sample · ${esc(s.name)}</a>`).join(' ');
}

function modelHtml(m, facts) {
    const picks = value('picks');
    const [first, ...rest] = m.source;
    const from = first ? `<span>from <a href="${repoUrl(first)}" target="_blank" rel="noopener" translate="no">${esc(first)}</a>${rest.length ? ` and ${esc(rest.length)} more` : ''}</span>` : '';
    return `<li class="row">
        <div class="head">
            <h3 class="name" translate="no"><a href="${repoUrl(m.id)}" target="_blank" rel="noopener">${esc(m.name)}</a></h3>
            ${picks && picks.keys.has(key(m.id)) ? '<a class="tag pick" href="#picks" title="In the picks on the organization card">Pick</a>' : ''}
        </div>
        <div class="meta">
            ${m.task ? `<span class="tag task">${esc(m.task)}</span>` : ''}
            ${m.formats.map(ext => `<span class="tag">.${esc(ext)}</span>`).join(' ')}
            ${m.gated ? '<span class="tag gated">Gated</span>' : ''}
            ${sampleTags([m.id])}
            ${from}
        </div>
        ${facts.length ? `<div class="facts">${facts.map(esc).join(' · ')}</div>` : ''}
    </li>`;
}
// What a row says under each list: the Hub's two counts under Trending, the creation date under Recently created.
const trendingRow = m => modelHtml(m, [m.likes ? plural(m.likes, 'like') : '', m.downloads ? `${plural(m.downloads, 'download')} last month` : ''].filter(Boolean));
const recentRow = m => modelHtml(m, [`created ${day(m.created)}`]);

function pickHtml(item) {
    const tags = sampleTags(item.repos);
    return `<li class="pick"><p>${inlineHtml(item.md)}</p>${tags ? `<div class="meta">${tags}</div>` : ''}</li>`;
}

function sampleHtml(s) {
    // A model's name as the model list spells it; as the table spells it when there is no model list.
    const models = value('models');
    const listed = id => (models ? (models.find(m => key(m.id) === key(id)) || {}).id : id);
    const links = s.models.map(listed).filter(Boolean).map(id => `<a href="${repoUrl(id)}" target="_blank" rel="noopener" translate="no">${esc(id.slice(id.indexOf('/') + 1))}</a>`).join(', ');
    return `<li class="row">
        <div class="head"><h4 class="name" translate="no"><a href="${esc(s.url)}" target="_blank" rel="noopener"><code>${esc(s.path)}</code></a></h4></div>
        ${s.task ? `<div class="what">${esc(s.task)}</div>` : ''}
        <div class="facts">${[s.platform ? esc(s.platform) : '', links ? `runs ${links}` : ''].filter(Boolean).join(' · ')}</div>
    </li>`;
}

function demoHtml(d) {
    return `<li class="row">
        <div class="head"><h4 class="name" translate="no"><a href="${repoUrl(d.id, 'spaces/')}" target="_blank" rel="noopener">${esc(d.title)}</a></h4></div>
        ${d.description ? `<div class="what">${esc(d.description)}</div>` : ''}
        <div class="facts">${[`<span translate="no">${esc(d.id)}</span>`, d.likes ? esc(plural(d.likes, 'like')) : ''].filter(Boolean).join(' · ')}</div>
    </li>`;
}

// The keys of the repos the page lists: both model lists, folded rows included, and the picks.
function listedKeys() {
    const models = value('models');
    const picks = value('picks');
    const keys = new Set(picks ? picks.keys : []);
    for (const list of models ? [trendingModels, recentModels] : []) {
        try {
            list(models).forEach(m => keys.add(key(m.id)));
        } catch (err) {
            // The list's own part says why it has no rows.
        }
    }
    return keys;
}

// One part of the page: its loading line while its source is being read, its notice with the reason when
// the source failed or the part could not be drawn, its rows otherwise. `fill` draws the rows.
function part(name, state, fill) {
    const clear = () => {
        for (const el of document.querySelectorAll(`#${name} .rows`)) el.innerHTML = '';
        for (const el of document.querySelectorAll(`#${name} .more, #${name} .aside`)) el.hidden = true;
    };
    $(`${name}Loading`).hidden = state !== null;
    $(`${name}Error`).hidden = true;
    clear();
    if (!state) return;
    try {
        if (!state.ok) throw new Error(state.why);
        fill(state.value);
    } catch (err) {
        const why = err && err.message ? err.message : String(err);
        if (state.ok && !logged.has(`${name}: ${why}`)) {
            logged.add(`${name}: ${why}`);
            console.error(name, err);
        }
        clear();
        $(`${name}ErrorWhy`).textContent = why;
        $(`${name}Error`).hidden = false;
    }
}

// A model list: its first rows, and the next ones behind a fold that says how many they are.
function fillList(name, list, row) {
    const first = list.slice(0, LIST_LENGTH).map(row).join('');
    const more = list.slice(LIST_LENGTH).map(row);
    $(`${name}List`).innerHTML = first;
    $(`${name}MoreList`).innerHTML = more.join('');
    $(`${name}MoreCount`).textContent = fmtCount(more.length);
    $(`${name}More`).hidden = more.length === 0;
}

function draw() {
    part('trending', got.models, models => {
        const list = trendingModels(models);
        fillList('trending', list, trendingRow);
        $('trendingEmpty').hidden = list.length > 0;
    });
    part('recent', got.models, models => fillList('recent', recentModels(models), recentRow));
    part('picks', got.picks, picks => {
        const rows = picks.items.map(pickHtml).join('');
        $('picksTitle').textContent = picks.heading;
        $('picksList').innerHTML = rows;
    });
    // The samples wait for the model list and the card as well, since which of them the page lists depends
    // on both. That the tables could not be read is shown at once.
    const samplesReady = got.samples && (!got.samples.ok || (got.models && got.picks));
    part('samples', samplesReady ? got.samples : null, ({ samples, failed }) => {
        const rows = samplesFor(samples, listedKeys()).map(sampleHtml);
        $('samplesList').innerHTML = rows.join('');
        $('samplesPartial').hidden = failed.length === 0;
        $('samplesPartialWhich').textContent = failed.map(f => `${f.dir}README.md`).join(', ');
        $('samplesEmpty').hidden = rows.length > 0;
    });
    part('demos', got.demos, demos => { $('demosList').innerHTML = demos.map(demoHtml).join(''); });
}

// Reads one source and redraws the page with what it gave, so no part waits for a source it does not show.
async function read(name, load) {
    try {
        got[name] = { ok: true, value: await load() };
    } catch (err) {
        console.error(name, err);
        got[name] = { ok: false, why: err && err.message ? err.message : String(err) };
    }
    draw();
}

function main() {
    draw();
    return Promise.all([read('models', loadModels), read('picks', loadPicks), read('samples', loadSamples), read('demos', loadDemos)]);
}

if (typeof document !== 'undefined') {
    main();
}

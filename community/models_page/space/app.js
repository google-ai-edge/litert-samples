// LiteRT Models: the litert-community model list from the Hugging Face API, joined with the
// benchmarks of the leaderboard in litert-samples and the recipes its models/README.md lists. All three
// are read when the page loads.

const HUB = "https://huggingface.co";
const HUB_MODELS_API = "https://huggingface.co/api/models";
export const MODELS_URL = HUB_MODELS_API + '?author=litert-community&limit=1000&full=true&cardData=true';
export const BOARD_URL = "https://google-ai-edge.github.io/litert-samples/benchmark/leaderboard/data/board.json";
export const RECIPES_URL = "https://google-ai-edge.github.io/litert-samples/models/README.md";
const RECIPES_DIR = "https://github.com/google-ai-edge/litert-samples/tree/main/models/";
const HOW_ROWS_URL = "https://github.com/google-ai-edge/litert-samples/tree/main/benchmark/leaderboard#how-a-row-is-made";
const TIMEOUT_MS = { models: 20000, board: 8000, recipes: 8000, files: 10000 };
const MAX_PAGES = 20;
const CARD_LINES = 3;
const TASK_CHIPS = 8;

export const FORMATS = ['tflite', 'litertlm', 'task'];
export const SORTS = [
    { id: 'updated', label: 'Recently updated' },
    { id: 'benchmarks', label: 'With benchmarks first' },
    { id: 'downloads', label: 'Most downloads' },
    { id: 'name', label: 'A–Z' },
];

// ---- formatting -------------------------------------------------------------------------------

export function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
export function fmt(v, digits = 2) {
    if (v == null || v === '' || !Number.isFinite(Number(v))) return '—';
    return Number(v).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
export function fmtBytes(n) {
    if (!Number.isFinite(n)) return '—';
    if (n < 1e6) return fmt(n / 1e3, 0) + ' kB';
    if (n < 1e9) return fmt(n / 1e6, 1) + ' MB';
    return fmt(n / 1e9, 2) + ' GB';
}
const day = iso => (typeof iso === 'string' ? iso.slice(0, 10) : '');
const text = v => (v == null || v === '' ? '—' : String(v));
const plural = (n, one) => `${n} ${one}${n === 1 ? '' : 's'}`;
const cmp = (x, y) => (x < y ? -1 : x > y ? 1 : 0);
const byLabel = (x, y) => x.localeCompare(y, 'en', { sensitivity: 'base' }) || cmp(x, y);
const repoPath = id => String(id).split('/').map(encodeURIComponent).join('/');
const treeUrl = id => `${HUB_MODELS_API}/${repoPath(id)}/tree/main?recursive=true`;
const repoUrl = (id, path = '') => esc(`${HUB}/${repoPath(id)}${path}`);
const REPO_ID = /^[A-Za-z0-9][\w.-]*\/(?=[\w.-]*\w)[\w.-]+$/;
const RECIPE_PATH = /^([A-Za-z0-9_-][A-Za-z0-9._-]*\/)*[A-Za-z0-9_-][A-Za-z0-9._-]*\/?$/;

// ---- data -------------------------------------------------------------------------------------

// The checkpoints a card's base_model field names, kept only as repo ids.
export function sourceModels(card) {
    const v = card && typeof card === 'object' ? card.base_model : null;
    const list = Array.isArray(v) ? v : [v];
    return [...new Set(list.filter(s => typeof s === 'string' && REPO_ID.test(s)))].slice(0, 8);
}

// The `Link: <url>; rel="next"` header the Hub sends when a listing has another page.
export function nextLink(header) {
    if (!header) return null;
    for (const part of header.split(',')) {
        const m = part.match(/<([^>]+)>\s*;\s*rel="?next"?/);
        if (m) return m[1];
    }
    return null;
}

// Public repos that hold at least one file besides the card and .gitattributes.
export function normalizeModels(list) {
    if (!Array.isArray(list)) throw new Error('the answer is not a model list');
    return list.filter(m => m && typeof m.id === 'string' && !m.private).map(m => {
        const names = Array.isArray(m.siblings) ? m.siblings.map(s => s && s.rfilename).filter(n => typeof n === 'string') : null;
        const files = Object.fromEntries(FORMATS.map(ext => [ext, []]));
        for (const name of names || []) {
            const ext = name.slice(name.lastIndexOf('.') + 1).toLowerCase();
            if (FORMATS.includes(ext)) files[ext].push(name);
        }
        return {
            id: m.id,
            name: m.id.includes('/') ? m.id.slice(m.id.indexOf('/') + 1) : m.id,
            task: typeof m.pipeline_tag === 'string' && m.pipeline_tag ? m.pipeline_tag : null,
            downloads: Number.isFinite(m.downloads) ? m.downloads : 0,
            updated: day(m.lastModified),
            gated: Boolean(m.gated),
            empty: names !== null && names.every(n => n === 'README.md' || n === '.gitattributes'),
            files,
            formats: FORMATS.filter(ext => files[ext].length),
            source: sourceModels(m.cardData),
            recipe: null,
            rows: [],
            lmRows: [],
        };
    }).filter(m => !m.empty);
}

// The "Model list" table of litert-samples/models/README.md: the recipe directory of each converted-weights
// repo. The columns are found by their header names. A row the page cannot read as one directory for a repo of
// the org (two directories, a path outside models/, a cell count that differs from the header's) is skipped and counted.
export function parseRecipes(text) {
    if (typeof text !== 'string') throw new Error('the answer is not text');
    const link = /\[[^\]]*\]\(([^)\s]+)\)/g;
    const cells = line => line.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
    const orgRepo = url => (url.match(/^https:\/\/huggingface\.co\/(litert-community\/[\w.-]+)\/?$/) || [])[1];
    const relative = url => url.replace(/^\.\//, '').replace(RECIPES_DIR, '').replace(/#.*$/, '');
    const recipes = new Map();
    let cols = null;
    let skipped = 0;
    for (const line of text.split('\n')) {
        if (!line.trim().startsWith('|')) { if (cols) break; continue; }
        const row = cells(line);
        if (!cols) {
            const names = row.map(c => c.toLowerCase());
            const recipe = names.indexOf('recipe');
            const weights = names.indexOf('converted weights');
            if (recipe >= 0 && weights >= 0) cols = { recipe, weights, width: row.length };
            continue;
        }
        if (row.every(c => /^:?-+:?$/.test(c))) continue;
        const named = [...line.matchAll(link)].some(m => orgRepo(m[1]));
        if (row.length !== cols.width) { if (named) skipped++; continue; }
        const paths = [...row[cols.recipe].matchAll(link)].map(m => relative(m[1])).filter(p => RECIPE_PATH.test(p));
        const repos = [...row[cols.weights].matchAll(link)].map(m => orgRepo(m[1])).filter(Boolean);
        if (paths.length !== 1) { if (repos.length) skipped++; continue; }
        const path = paths[0].replace(/\/?$/, '/');
        for (const repo of repos) if (!recipes.has(repo)) recipes.set(repo, { path, url: RECIPES_DIR + path });
    }
    if (!cols) throw new Error('no model list in the README');
    return { recipes, skipped };
}

// Attaches each model's recipe. Recipes of a repo the list does not have are returned as orphans.
export function joinRecipes(models, recipes) {
    const ids = new Set(models.map(m => m.id));
    for (const m of models) m.recipe = recipes.get(m.id) || null;
    return { orphans: [...recipes.keys()].filter(id => !ids.has(id)) };
}

const isRow = r => r && typeof r === 'object' && r.status === 'measured'
    && ['model', 'file', 'platform', 'device', 'accelerator'].every(k => typeof r[k] === 'string' && r[k]);

// Groups the board's benchmarks by model repo. Throws when the file is not a board.
export function indexBoard(board) {
    if (!board || !Array.isArray(board.rows)) throw new Error('board.json has no rows');
    const rows = board.rows.filter(r => isRow(r) && r.latency_ms && typeof r.latency_ms === 'object');
    const lmRows = (Array.isArray(board.lm_rows) ? board.lm_rows : []).filter(r => isRow(r) && r.metrics && typeof r.metrics === 'object');
    const all = [...rows, ...lmRows];
    const platforms = (Array.isArray(board.platforms) ? board.platforms : [])
        .filter(p => p && all.some(r => r.platform === p.id)).map(p => ({ id: p.id, name: p.name || p.id }));
    for (const id of new Set(all.map(r => r.platform))) if (!platforms.some(p => p.id === id)) platforms.push({ id, name: id });
    const index = {
        byModel: new Map(),
        platforms,
        devices: [...new Map(all.map(r => [r.device, { name: r.device, platform: r.platform }])).values()].sort((a, b) => byLabel(a.name, b.name)),
        accelerators: [...new Set(all.map(r => r.accelerator))].sort(),
        count: all.length,
        newest: all.map(r => day(r.date)).filter(Boolean).sort().pop() || day(board.generated_at),
    };
    const slot = id => index.byModel.get(id) || index.byModel.set(id, { rows: [], lmRows: [] }).get(id);
    orderRows(rows, index).forEach(r => slot(r.model).rows.push(r));
    orderRows(lmRows, index).forEach(r => slot(r.model).lmRows.push(r));
    return index;
}

// A fixed reading order: platform as the board lists it, then device, accelerator and file.
export function orderRows(rows, index) {
    const platform = id => { const i = index.platforms.findIndex(p => p.id === id); return i < 0 ? index.platforms.length : i; };
    return [...rows].sort((a, b) => platform(a.platform) - platform(b.platform) || byLabel(a.device, b.device)
        || cmp(a.accelerator, b.accelerator) || cmp(a.file, b.file));
}

// Attaches each model's benchmarks. Benchmarks of a repo the list does not have are returned as orphans.
export function joinRows(models, index) {
    const ids = new Set(models.map(m => m.id));
    for (const m of models) {
        const hit = index.byModel.get(m.id);
        m.rows = hit ? hit.rows : [];
        m.lmRows = hit ? hit.lmRows : [];
    }
    return { orphans: [...index.byModel.keys()].filter(id => !ids.has(id)) };
}

export const hasBenchmarks = m => m.rows.length + m.lmRows.length > 0;
const rowMatches = (r, f) => (!f.platform || r.platform === f.platform) && (!f.device || r.device === f.device) && (!f.accelerator || r.accelerator === f.accelerator);
const narrowsRows = f => Boolean(f.platform || f.device || f.accelerator);
export const matchingRows = (m, f) => ({ rows: m.rows.filter(r => rowMatches(r, f)), lmRows: m.lmRows.filter(r => rowMatches(r, f)) });

export function filterModels(models, f) {
    const q = (f.q || '').trim().toLowerCase();
    return models.filter(m => {
        if (f.task && (f.task === '(other)' ? m.task !== null : m.task !== f.task)) return false;
        if (f.format && !m.formats.includes(f.format)) return false;
        if (f.recipe && !m.recipe) return false;
        if (q && !m.name.toLowerCase().includes(q)) return false;
        if (f.benchmarks || narrowsRows(f)) {
            const hit = matchingRows(m, f);
            if (hit.rows.length + hit.lmRows.length === 0) return false;
        }
        return true;
    });
}

export function sortModels(models, sort) {
    const byName = (a, b) => byLabel(a.name, b.name);
    const byDownloads = (a, b) => b.downloads - a.downloads || byName(a, b);
    const order = {
        benchmarks: (a, b) => Number(hasBenchmarks(b)) - Number(hasBenchmarks(a)) || byDownloads(a, b),
        downloads: byDownloads,
        updated: (a, b) => cmp(b.updated, a.updated) || byName(a, b),
        name: byName,
    }[sort] || byName;
    return [...models].sort(order);
}

export function taskCounts(models) {
    const counts = new Map();
    for (const m of models) counts.set(m.task || '(other)', (counts.get(m.task || '(other)') || 0) + 1);
    return [...counts.entries()].sort((a, b) => (a[0] === '(other)') - (b[0] === '(other)') || b[1] - a[1] || cmp(a[0], b[0]));
}

// Device options under the chosen platform.
export function deviceOptions(index, platform) {
    return index.devices.filter(d => !platform || d.platform === platform).map(d => d.name);
}

// What a card shows of a model's benchmarks: the file (for LiteRT-LM, the file and token counts) with the
// most of them, one line per device, and on a line one number per accelerator, the newest when there are several.
export function cardSummary(rows, kind, limit = CARD_LINES) {
    if (!rows.length) return null;
    const cond = r => (r.conditions && typeof r.conditions === 'object' ? r.conditions : {});
    const key = r => (kind === 'lm' ? [r.file, cond(r).prefill_tokens, cond(r).decode_tokens, cond(r).max_num_tokens].join('\n') : r.file);
    const counts = new Map();
    rows.forEach(r => counts.set(key(r), (counts.get(key(r)) || 0) + 1));
    const best = [...counts.entries()].sort((a, b) => b[1] - a[1] || cmp(a[0], b[0]))[0][0];
    const mine = rows.filter(r => key(r) === best);
    const lines = [];
    for (const r of mine) {
        const id = r.device_id || r.device;
        let line = lines.find(l => l.id === id);
        if (!line) lines.push(line = { id, device: r.device, os: typeof r.os === 'string' ? r.os : '', cells: [] });
        const value = kind === 'lm' ? r.metrics.decode_tok_s : r.latency_ms.median;
        const cell = line.cells.find(c => c.accelerator === r.accelerator);
        if (!cell) line.cells.push({ accelerator: r.accelerator, value, date: day(r.date) });
        else if (day(r.date) > cell.date) Object.assign(cell, { value, date: day(r.date) });
    }
    const shown = lines.slice(0, limit);
    const twice = name => shown.filter(l => l.device === name).length > 1;
    const { prefill_tokens: prefill, decode_tokens: decode } = cond(mine[0]);
    return {
        file: mine[0].file,
        tokens: kind === 'lm' && Number.isFinite(prefill) && Number.isFinite(decode) ? { prefill, decode } : null,
        lines: shown.map(l => ({
            device: twice(l.device) && l.os ? `${l.device}, ${l.os}` : l.device,
            date: l.cells.map(c => c.date).sort().pop() || '',
            cells: l.cells.map(({ accelerator, value }) => ({ accelerator, value })),
        })),
        shown: shown.reduce((n, l) => n + l.cells.length, 0),
    };
}

// ---- loading ----------------------------------------------------------------------------------

async function getText(url, timeoutMs, accept) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    try {
        const res = await fetch(url, { signal: ctl.signal, headers: { Accept: accept } });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return { body: await res.text(), link: res.headers.get('Link') };
    } catch (err) {
        throw new Error(err.name === 'AbortError' ? `no answer in ${timeoutMs / 1000} s` : err.message || String(err));
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
    for (let page = 0; url && page < MAX_PAGES; page++) {
        const { data, link } = await get(url);
        if (!Array.isArray(data)) throw new Error('the answer is not a model list');
        all.push(...data);
        url = nextLink(link);
    }
    const models = normalizeModels(all);
    if (!models.length) throw new Error('the list came back empty');
    return models;
}

async function loadBoard() {
    const index = indexBoard((await getJson(BOARD_URL, TIMEOUT_MS.board)).data);
    if (!index.count) throw new Error('board.json holds no benchmark this page can read');
    return index;
}

async function loadRecipes() {
    const { recipes, skipped } = parseRecipes((await getText(RECIPES_URL, TIMEOUT_MS.recipes, 'text/markdown, text/plain')).body);
    if (skipped) console.warn(`${skipped} row(s) of the model list could not be read as one directory for a repo of the org and are left out`);
    return recipes;
}

// ---- page -------------------------------------------------------------------------------------

const $ = id => document.getElementById(id);
const state = { task: '', format: '', benchmarks: false, recipe: false, platform: '', device: '', accelerator: '', q: '', sort: 'updated', open: null, moreTasks: false };
let MODELS = [];
let INDEX = null;
let RECIPES = null;
const TREES = new Map();

const chip = (group, value, label, count, on) =>
    `<button type="button" class="chip${on ? ' on' : ''}" data-group="${esc(group)}" data-value="${esc(value)}" aria-pressed="${on}">${esc(label)}${count == null ? '' : ` <span class="n">${esc(count)}</span>`}</button>`;
const option = (value, label, on) => `<option value="${esc(value)}"${on ? ' selected' : ''}>${esc(label)}</option>`;

function drawFilters() {
    const tasks = taskCounts(MODELS);
    const shown = state.moreTasks ? tasks : tasks.slice(0, TASK_CHIPS);
    if (state.task && !shown.some(([t]) => t === state.task)) shown.push(tasks.find(([t]) => t === state.task));
    $('fTask').innerHTML = chip('task', '', 'Any', null, !state.task)
        + shown.map(([t, n]) => chip('task', t, t === '(other)' ? 'Other' : t, n, state.task === t)).join('')
        + (tasks.length > TASK_CHIPS ? `<button type="button" class="chip more" id="moreTasks" aria-expanded="${state.moreTasks}">${state.moreTasks ? 'Fewer tasks' : `More tasks (${tasks.length - TASK_CHIPS})`}</button>` : '');
    $('fFormat').innerHTML = chip('format', '', 'Any', null, !state.format)
        + FORMATS.map(ext => chip('format', ext, `.${ext}`, MODELS.filter(m => m.formats.includes(ext)).length, state.format === ext)).join('');
    $('fBenchmarks').innerHTML = chip('benchmarks', '', 'Any', null, !state.benchmarks)
        + chip('benchmarks', '1', 'With benchmarks', INDEX ? MODELS.filter(hasBenchmarks).length : null, state.benchmarks);
    $('fBenchmarks').querySelectorAll('button')[1].disabled = !INDEX;
    $('fRows').hidden = !INDEX;
    $('fRecipe').innerHTML = chip('recipe', '', 'Any', null, !state.recipe)
        + chip('recipe', '1', 'With a recipe', RECIPES ? MODELS.filter(m => m.recipe).length : null, state.recipe);
    $('fRecipe').querySelectorAll('button')[1].disabled = !RECIPES;
    if (INDEX) {
        $('fPlatform').innerHTML = option('', 'All', !state.platform) + INDEX.platforms.map(p => option(p.id, p.name, state.platform === p.id)).join('');
        $('fDevice').innerHTML = option('', 'All', !state.device) + deviceOptions(INDEX, state.platform).map(d => option(d, d, state.device === d)).join('');
        $('fAccel').innerHTML = option('', 'All', !state.accelerator) + INDEX.accelerators.map(a => option(a, a.toUpperCase(), state.accelerator === a)).join('');
    }
    $('fSort').innerHTML = SORTS.map(s => option(s.id, s.label, state.sort === s.id)).join('');
}

function summaryHtml(s, kind) {
    if (!s) return '';
    const unit = kind === 'lm' ? 'tok/s' : 'ms';
    const what = kind === 'lm'
        ? `median decode tokens/s${s.tokens ? ` · ${esc(s.tokens.prefill)} prefill, ${esc(s.tokens.decode)} decode tokens` : ''}`
        : 'median latency';
    const lines = s.lines.map(l => `<li><span class="device">${esc(l.device)}</span> ${l.cells.map(c =>
        `<span class="cell"><span class="acc">${esc(c.accelerator.toUpperCase())}</span> <b>${fmt(c.value, kind === 'lm' ? 1 : 2)}</b> ${unit}</span>`).join(' ')} <span class="date">${esc(l.date)}</span></li>`).join('');
    return `<div class="what"><code>${esc(s.file)}</code> · ${what}</div><ul>${lines}</ul>`;
}

// The numbers a card shows up front; the details hold every benchmark of the model, so the count of the
// rest is taken against all of them, also while the list is narrowed to a platform, device or accelerator.
function benchHtml(m, open) {
    const hit = matchingRows(m, state);
    const a = cardSummary(hit.rows, 'litert');
    const b = cardSummary(hit.lmRows, 'lm');
    if (!a && !b) return '';
    const more = m.rows.length + m.lmRows.length - (a ? a.shown : 0) - (b ? b.shown : 0);
    const button = more > 0 && !open ? `<button type="button" class="link" data-id="${esc(m.id)}">+ ${esc(plural(more, 'more benchmark'))}</button>` : '';
    return `<div class="bench">${summaryHtml(a, 'litert')}${summaryHtml(b, 'lm')}${button}</div>`;
}

// The run behind a row: the binary it ran (the bucket object, or the source tag for iOS; for a `latest`
// LiteRT-LM binary its sha256), the runs it completed, and the file the numbers came from. benchmark_model's
// warm-up runs come before the counted runs; a LiteRT-LM run's warm-up iteration is one of its iterations.
function runCell(r, kind) {
    const [object, sha] = (typeof r.binary === 'string' ? r.binary : '').split(/,\s*sha256\s+/);
    const n = v => (Number.isFinite(v) ? v : null);
    const done = kind === 'lm' ? n(r.conditions && r.conditions.iterations) : n(r.runs);
    const warm = kind === 'lm' ? n(r.conditions && r.conditions.warmup_iterations) : n(r.warmup_runs);
    const counts = done == null ? '' : plural(done, kind === 'lm' ? 'iteration' : 'run')
        + (warm == null ? '' : kind === 'lm' ? `, ${warm} of them warm-up` : ` after ${warm} warm-up`);
    const source = typeof r.source === 'string' && r.source ? `from ${r.source === 'log' ? 'the log' : r.source}` : '';
    const line = [counts, source].filter(Boolean).join(' · ');
    return `<td class="run">${object ? `<code>${esc(object)}</code>` : '—'} ${/^[0-9a-f]{12,}$/i.test(sha || '') ? `<div class="sub">sha256 ${esc(sha.slice(0, 12))}…</div> ` : ''}${line ? `<div class="sub">${esc(line)}</div>` : ''}</td>`;
}

function rowsTable(rows) {
    if (!rows.length) return '';
    const body = rows.map(r => `<tr>
        <td class="file"><code>${esc(r.file)}</code><div class="sub">${r.model_size_mb == null ? '' : fmt(r.model_size_mb, 1) + ' MB'}</div></td>
        <td>${esc(r.device)}<div class="sub">${esc(r.os || '')}</div></td>
        <td>${esc(r.accelerator.toUpperCase())}<div class="sub">${esc(r.delegate || '')}</div>${r.nodes_delegated == null ? '' : `<div class="sub">${esc(r.nodes_delegated)}/${esc(text(r.nodes_total))} nodes delegated</div>`}</td>
        <td class="num key">${fmt(r.latency_ms.median)}</td><td class="num">${fmt(r.latency_ms.p95)}</td><td class="num">${fmt(r.latency_ms.init)}</td>
        <td class="num">${fmt(r.memory_mb && r.memory_mb.overall_footprint, 1)}</td>
        <td>${esc(text(r.runtime_version))}<div class="sub">${esc(day(r.date))}</div></td>${runCell(r, 'litert')}</tr>`).join('');
    return `<h4>LiteRT benchmarks <span class="sub">benchmark_model</span></h4><div class="scroll"><table>
        <thead><tr><th>File</th><th>Device</th><th>Accelerator</th><th class="num">Median <span class="unit">ms</span></th><th class="num">p95 <span class="unit">ms</span></th><th class="num">Init <span class="unit">ms</span></th><th class="num">Footprint <span class="unit">MB</span></th><th>Runtime, date</th><th>Run</th></tr></thead>
        <tbody>${body}</tbody></table></div>`;
}

function lmRowsTable(rows) {
    if (!rows.length) return '';
    const tokens = (r, k) => (r.conditions && Number.isFinite(r.conditions[k]) ? `${esc(r.conditions[k])} tokens` : '');
    const body = rows.map(r => `<tr>
        <td class="file"><code>${esc(r.file)}</code><div class="sub">${r.model_size_mb == null ? '' : fmt(r.model_size_mb, 1) + ' MB'}</div></td>
        <td>${esc(r.device)}<div class="sub">${esc(r.os || '')}</div></td>
        <td>${esc(r.accelerator.toUpperCase())}<div class="sub">${esc(r.delegate || '')}</div></td>
        <td class="num">${fmt(r.metrics.prefill_tok_s, 1)}<div class="sub">${tokens(r, 'prefill_tokens')}</div></td>
        <td class="num key">${fmt(r.metrics.decode_tok_s, 1)}<div class="sub">${tokens(r, 'decode_tokens')}</div></td>
        <td class="num">${fmt(r.metrics.ttft_s, 3)}</td><td class="num">${fmt(r.metrics.init_total_ms, 0)}</td>
        <td>${esc(text(r.runtime_version))}<div class="sub">${esc(day(r.date))}</div></td>${runCell(r, 'lm')}</tr>`).join('');
    return `<h4>LiteRT-LM benchmarks <span class="sub">LiteRT-LM benchmark binary; prefill, decode and first token are medians over the iterations after the warm-up, Init is the one engine creation</span></h4><div class="scroll"><table>
        <thead><tr><th>File</th><th>Device</th><th>Backend</th><th class="num">Prefill <span class="unit">tok/s</span></th><th class="num">Decode <span class="unit">tok/s</span></th><th class="num">First token <span class="unit">s</span></th><th class="num">Init <span class="unit">ms</span></th><th>Runtime, date</th><th>Run</th></tr></thead>
        <tbody>${body}</tbody></table></div>`;
}

function filesHtml(m) {
    const tree = TREES.get(m.id);
    const names = FORMATS.flatMap(ext => m.files[ext]);
    if (!names.length) return '';
    const size = name => {
        if (tree === 'failed') return '';
        if (!(tree instanceof Map)) return '…';
        return esc(fmtBytes(tree.get(name)));
    };
    const note = tree === 'failed' ? `<p class="sub">File sizes could not be loaded. They are on the <a href="${repoUrl(m.id, '/tree/main')}" target="_blank" rel="noopener">Files tab</a>.</p>` : '';
    return `<h4>Model files</h4><ul class="files">${names.map(n => `<li><code>${esc(n)}</code> <span class="size">${size(n)}</span></li>`).join('')}</ul>${note}`;
}

// What the model links to: its card and files, the checkpoint it was converted from, its recipe in
// litert-samples, and how its benchmarks were made. A line whose fact is missing is not shown.
function resourcesHtml(m) {
    const ext = (url, label) => `<a href="${url}" target="_blank" rel="noopener">${label}</a>`;
    const items = [
        `${ext(repoUrl(m.id), 'Model card')} · ${ext(repoUrl(m.id, '/tree/main'), 'Files')}`,
        m.source.length ? `Source model: ${m.source.map(s => ext(esc(`${HUB}/${repoPath(s)}`), esc(s))).join(', ')}` : '',
        m.recipe ? `Recipe: ${ext(esc(m.recipe.url), `litert-samples/models/${esc(m.recipe.path)}`)}` : '',
        hasBenchmarks(m) ? ext(HOW_ROWS_URL, 'How these benchmarks were made') : '',
    ].filter(Boolean);
    return `<h4>Resources</h4><ul class="resources">${items.map(i => `<li>${i}</li>`).join('')}</ul>`;
}

function detailsHtml(m) {
    return `<div class="details">
        ${rowsTable(m.rows)}${lmRowsTable(m.lmRows)}<div class="filesbox">${filesHtml(m)}</div>
        ${resourcesHtml(m)}
    </div>`;
}

function cardHtml(m) {
    const open = state.open === m.id;
    const formats = m.formats.map(ext => `<span class="tag">${esc(plural(m.files[ext].length, `.${ext} file`))}</span>`).join(' ');
    return `<li class="card" data-id="${esc(m.id)}">
        <div class="head">
            <h3 class="name"><a href="${repoUrl(m.id)}" target="_blank" rel="noopener">${esc(m.name)}</a></h3>
            <button type="button" class="toggle" data-id="${esc(m.id)}" aria-expanded="${open}" aria-label="${open ? 'Close details' : 'Details'}, ${esc(m.name)}">${open ? 'Close' : 'Details'}</button>
        </div>
        <div class="meta">
            ${m.task ? `<span class="tag task">${esc(m.task)}</span>` : ''} ${formats}
            ${m.gated ? '<span class="tag gated">Gated · request access on the model page</span>' : ''}
            ${m.recipe ? `<a class="tag recipe" href="${esc(m.recipe.url)}" target="_blank" rel="noopener">Recipe · litert-samples</a>` : ''}
            ${m.updated ? `<span>updated ${esc(m.updated)}</span>` : ''}
        </div>
        ${benchHtml(m, open)}
        ${open ? detailsHtml(m) : ''}
    </li>`;
}

function drawList() {
    const list = sortModels(filterModels(MODELS, state), state.sort);
    $('count').textContent = `${plural(list.length, 'model')}${list.length === MODELS.length ? '' : ` of ${MODELS.length}`}`;
    $('list').innerHTML = list.map(cardHtml).join('');
    $('empty').hidden = list.length > 0;
}

// A redraw replaces the buttons, so the one that had the focus is found again by what it is.
function keepFocus(redraw) {
    const el = document.activeElement;
    let key = null;
    if (el && el.id) key = `#${CSS.escape(el.id)}`;
    else if (el && el.dataset && el.dataset.group != null) key = `#filters button[data-group="${CSS.escape(el.dataset.group)}"][data-value="${CSS.escape(el.dataset.value)}"]`;
    redraw();
    const again = key && document.querySelector(key);
    if (again && again !== document.activeElement && !again.disabled && again.offsetParent !== null) again.focus();
}

function draw() {
    keepFocus(() => { drawFilters(); drawList(); });
}

function cardOf(id) {
    return [...document.querySelectorAll('#list .card')].find(c => c.dataset.id === id) || null;
}

async function toggleDetails(id) {
    state.open = state.open === id ? null : id;
    drawList();
    const card = cardOf(id);
    if (card) card.querySelector('button.toggle').focus();
    if (!state.open || TREES.get(id) instanceof Map || TREES.get(id) === 'loading') return;
    TREES.set(id, 'loading');
    try {
        const { data } = await getJson(treeUrl(id), TIMEOUT_MS.files);
        if (!Array.isArray(data)) throw new Error('unexpected answer');
        TREES.set(id, new Map(data.filter(f => f && f.type === 'file').map(f => [f.path, f.size])));
    } catch (err) {
        console.error('file sizes', id, err);
        TREES.set(id, 'failed');
    }
    const box = state.open === id && cardOf(id) && cardOf(id).querySelector('.filesbox');
    if (box) box.innerHTML = filesHtml(MODELS.find(m => m.id === id));
}

function clearFilters() {
    Object.assign(state, { task: '', format: '', benchmarks: false, recipe: false, platform: '', device: '', accelerator: '', q: '' });
    $('fName').value = '';
    draw();
}

// An error after the first draw (in a handler) is shown the same way as one before it.
function stopOn(err) {
    console.error(err);
    $('loading').hidden = true;
    $('app').hidden = true;
    $('pageErrorWhy').textContent = err && err.message ? err.message : String(err);
    $('pageError').hidden = false;
}
const guarded = fn => (...args) => { try { const out = fn(...args); if (out && typeof out.catch === 'function') out.catch(stopOn); } catch (err) { stopOn(err); } };

function wire() {
    $('filters').addEventListener('click', guarded(e => {
        const b = e.target.closest('button');
        if (!b) return;
        if (b.id === 'filtersToggle') return void b.setAttribute('aria-expanded', String($('filters').classList.toggle('open')));
        if (b.id === 'moreTasks') state.moreTasks = !state.moreTasks;
        else if (b.id === 'clear') return clearFilters();
        else if (b.dataset.group === 'benchmarks') state.benchmarks = b.dataset.value === '1';
        else if (b.dataset.group === 'recipe') state.recipe = b.dataset.value === '1';
        else if (b.dataset.group) state[b.dataset.group] = b.dataset.value;
        else return;
        draw();
    }));
    $('fPlatform').addEventListener('change', guarded(e => {
        state.platform = e.target.value;
        if (state.device && !deviceOptions(INDEX, state.platform).includes(state.device)) state.device = '';
        draw();
    }));
    $('fDevice').addEventListener('change', guarded(e => { state.device = e.target.value; draw(); }));
    $('fAccel').addEventListener('change', guarded(e => { state.accelerator = e.target.value; draw(); }));
    $('fSort').addEventListener('change', guarded(e => { state.sort = e.target.value; drawList(); }));
    $('fName').addEventListener('input', guarded(e => { state.q = e.target.value; drawList(); }));
    $('emptyClear').addEventListener('click', guarded(() => { clearFilters(); $('fName').focus(); }));
    $('list').addEventListener('click', guarded(e => {
        const b = e.target.closest('button[data-id]');
        return b ? toggleDetails(b.dataset.id) : undefined;
    }));
}

async function main() {
    const [models, board, recipes] = await Promise.allSettled([loadModels(), loadBoard(), loadRecipes()]);
    $('loading').hidden = true;
    if (models.status === 'rejected') {
        console.error('model list', models.reason);
        $('loadErrorWhy').textContent = models.reason.message;
        $('loadError').hidden = false;
        return;
    }
    MODELS = models.value;
    if (board.status === 'fulfilled') {
        INDEX = board.value;
        const { orphans } = joinRows(MODELS, INDEX);
        if (orphans.length) console.warn('benchmarks of repos the model list does not have:', orphans);
    } else {
        console.error('board.json', board.reason);
        $('boardError').hidden = false;
    }
    if (recipes.status === 'fulfilled') {
        RECIPES = recipes.value;
        const { orphans } = joinRecipes(MODELS, RECIPES);
        if (orphans.length) console.warn('recipes of repos the model list does not have:', orphans);
    } else {
        console.error('recipes', recipes.reason);
        $('recipesError').hidden = false;
    }
    $('stats').textContent = plural(MODELS.length, 'model') + (INDEX && INDEX.newest ? ` · newest benchmark ${INDEX.newest}` : '');
    $('app').hidden = false;
    wire();
    draw();
}

if (typeof document !== 'undefined') {
    main().catch(stopOn);
}

// node test/live_check.mjs — reads the live sources the page reads and checks what the page assumes about them.
// Exit status: 0 when every check holds, 1 when a source no longer looks the way the page assumes, 2 when a
// source could not be reached.
import assert from 'node:assert/strict';
import {
    MODELS_URL, CARD_URL, SAMPLE_INDEXES, DEMOS_URL, LIST_LENGTH, nextLink, normalizeModels, trendingModels, recentModels,
    parsePicks, parseSamples, samplesByModel, samplesFor, parseDemos,
} from '../space/app.js';

const MAIN = "https://raw.githubusercontent.com/google-ai-edge/litert-samples/main/";
class Unreachable extends Error {}
const ORIGIN = 'https://example.static.hf.space';
// A source as a page on that origin gets it: an answer of 200 without a redirect (the page's CSP names the
// address itself), open to that origin.
const fetchAs = async (url, accept) => {
    const res = await fetch(url, { headers: { Accept: accept, Origin: ORIGIN }, redirect: 'manual' }).catch(err => { throw new Unreachable(`${url}: ${err.message}`); });
    assert.equal(res.status, 200, `${url} -> ${res.status}`);
    assert.ok(['*', ORIGIN].includes(res.headers.get('access-control-allow-origin')), `${url}: not open to ${ORIGIN} (access-control-allow-origin: ${res.headers.get('access-control-allow-origin')})`);
    return res;
};
const getJson = async url => {
    const res = await fetchAs(url, 'application/json');
    return { data: await res.json(), link: res.headers.get('Link'), exposed: res.headers.get('access-control-expose-headers') || '' };
};
const getText = async url => (await fetchAs(url, 'text/markdown, text/plain')).text();

async function check() {
    const one = await getJson(MODELS_URL);
    assert.equal(nextLink(one.link), null, 'the org no longer fits one page: check the page follows Link in a browser');
    assert.match(one.exposed, /(^|,\s*)Link(,|$)/, 'the Hub does not expose Link to browsers');
    assert.ok(one.data.every(m => Number.isFinite(m.trendingScore)), 'a model without a trending score: check expand=trendingScore still works');
    assert.ok(one.data.every(m => Array.isArray(m.siblings) && typeof m.createdAt === 'string'), 'a model without its file list or creation date');
    const scores = one.data.map(m => m.trendingScore);
    assert.ok(scores.every((s, i) => i === 0 || s <= scores[i - 1]), 'the listing is not in the order of the trending score');
    assert.ok(one.data.some(m => m.cardData && m.cardData.base_model), 'the listing carries no card metadata: check expand=cardData still works');

    const models = normalizeModels(one.data);
    const trending = trendingModels(models);
    const recent = recentModels(models);
    // The Hub's own Recently created order, to compare with the page's: the same creation dates in the same places.
    const created = normalizeModels((await getJson(MODELS_URL.replace('sort=trendingScore', 'sort=createdAt'))).data);
    assert.deepEqual(recent.map(m => m.created), created.slice(0, recent.length).map(m => m.created), 'the newest models differ from the Hub\'s Recently created order');

    const card = await getText(CARD_URL);
    const picks = parsePicks(card);
    assert.equal(picks.dropped, 0, 'the picks section has something the page does not show: a line that is in no list item, an item past the twelfth, or an item cut short');
    assert.equal(picks.unlinked, 0, 'the picks link a site the page shows as text');
    const picked = [...new Set(picks.items.flatMap(i => i.repos).map(id => id.toLowerCase()))];
    const ids = new Set(models.map(m => m.id.toLowerCase()));
    assert.deepEqual(picked.filter(id => !ids.has(id)), [], 'the picks link repos the org list does not have');

    const samples = [];
    for (const { url, dir } of SAMPLE_INDEXES) {
        const text = await getText(url);
        const table = parseSamples(text, dir);
        assert.ok(table.samples.length > 0, `${dir}README.md names no model of the org`);
        assert.equal(table.skipped, 0, `${dir}README.md has rows the page cannot read as one sample directory`);
        samples.push(...table.samples);
        const main = await fetch(MAIN + dir + 'README.md').then(r => (r.ok ? r.text() : null), () => null);
        if (main !== null && main !== text) console.log(`note: the GitHub Pages copy of ${dir}README.md differs from main (Pages follows a merge by some minutes)`);
    }
    const byModel = samplesByModel(samples);
    const listed = new Set([...trending, ...recent].map(m => m.id.toLowerCase()).concat(picked));
    const demos = parseDemos((await getJson(DEMOS_URL)).data);
    assert.ok(demos.length > 0, 'the Demos collection lists no Space');

    console.log(`${one.data.length} repos from the API, ${models.length} listed, ${one.data.length - models.length} without files left out`);
    console.log(`trending: ${trending.length} models with a score above zero, the first ${Math.min(trending.length, LIST_LENGTH)} unfolded (${trending.slice(0, 3).map(m => m.name).join(', ')}, …)`);
    console.log(`recently created: the newest ${recent.length}, ${recent[0].created.slice(0, 10)} back to ${recent.at(-1).created.slice(0, 10)}, the first ${LIST_LENGTH} unfolded (${recent.slice(0, 3).map(m => m.name).join(', ')}, …)`);
    console.log(`picks: "${picks.heading}", ${picks.items.length} items linking ${picked.length} repos of the org, ${picked.filter(id => trending.concat(recent).some(m => m.id.toLowerCase() === id)).length} of them on a model list`);
    console.log(`samples: ${samples.length} rows that name a model of the org, ${samplesFor(samples, listed).length} of them run a model the page lists; ${byModel.size} repos with a sample`);
    console.log(`demos: ${demos.map(d => d.id).join(', ')}`);
    console.log('PASS live_check');
}

try {
    await check();
} catch (err) {
    console.error(err instanceof Unreachable ? `UNREACHABLE ${err.message}` : err);
    process.exit(err instanceof Unreachable ? 2 : 1);
}

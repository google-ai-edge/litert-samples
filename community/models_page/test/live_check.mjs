// node test/live_check.mjs — reads the three live sources the page reads and checks what the page assumes about them.
import assert from 'node:assert/strict';
import { MODELS_URL, BOARD_URL, RECIPES_URL, nextLink, normalizeModels, indexBoard, joinRows, hasBenchmarks, parseRecipes, joinRecipes } from '../space/app.js';

const fetchAs = async (url, accept) => {
    const res = await fetch(url, { headers: { Accept: accept, Origin: 'https://example.static.hf.space' } });
    assert.equal(res.status, 200, `${url} -> ${res.status}`);
    assert.ok(res.headers.get('access-control-allow-origin'), `${url}: no CORS header`);
    return res;
};
const get = async url => {
    const res = await fetchAs(url, 'application/json');
    return { data: await res.json(), link: res.headers.get('Link'), exposed: res.headers.get('access-control-expose-headers') || '' };
};

const one = await get(MODELS_URL);
assert.equal(nextLink(one.link), null, 'the org no longer fits one page: check the page follows Link in a browser');
assert.match(one.exposed, /(^|,\s*)Link(,|$)/, 'the Hub does not expose Link to browsers');

const paged = [];
for (let url = MODELS_URL.replace('limit=1000', 'limit=100'); url;) {
    const page = await get(url);
    paged.push(...page.data);
    url = nextLink(page.link);
}
assert.equal(paged.length, one.data.length, 'paging through Link gives another count than one call');

const models = normalizeModels(one.data);
assert.ok(one.data.some(m => m.cardData && m.cardData.base_model), 'the listing carries no card metadata: check cardData=true still works');
const board = (await get(BOARD_URL)).data;
const index = indexBoard(board);
const { orphans } = joinRows(models, index);
assert.deepEqual(orphans, [], 'benchmarks of repos the org list does not have');
assert.equal(index.count, board.rows.length + (board.lm_rows || []).length, 'the page drops benchmarks the board holds');

const { recipes, skipped } = parseRecipes(await (await fetchAs(RECIPES_URL, 'text/markdown, text/plain')).text());
assert.ok(recipes.size > 0, 'the model list names no litert-community repo');
const joinedRecipes = joinRecipes(models, recipes);
assert.deepEqual(joinedRecipes.orphans, [], 'recipes of repos the org list does not have');

console.log(`${one.data.length} repos from the API, ${models.length} listed, ${one.data.length - models.length} without files left out, ${models.filter(m => m.source.length).length} with a source model`);
console.log(`${index.count} benchmarks on ${models.filter(hasBenchmarks).length} models, the newest dated ${index.newest}; platforms ${index.platforms.map(p => p.id).join(', ')}; accelerators ${index.accelerators.join(', ')}`);
console.log(`${recipes.size} recipes on ${models.filter(m => m.recipe).length} models (${[...recipes.keys()].map(k => k.split('/')[1]).join(', ')}), ${skipped} row(s) the page could not read as one directory for a repo of the org`);
console.log('PASS live_check');

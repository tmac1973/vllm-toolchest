// Unit tests for the filtering logic in web/static/log-panel.js.
//
// log-panel.js is a plain browser script: it registers DOM listeners at load
// and exports nothing. It is run here in a vm context with a stub document,
// which leaves its top-level functions as globals of that context. Only the
// pure parts are tested: which lines a panel shows, and that hiding lines
// never loses them.

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const src = fs.readFileSync(path.join(__dirname, '..', 'static', 'log-panel.js'), 'utf8');
const ctx = vm.createContext({
    document: { addEventListener() {} },
});
vm.runInContext(src, ctx, { filename: 'log-panel.js' });
const { visibleLog, logPanelAppend, renderLog } = ctx;

let fails = 0;
function ok(cond, msg) {
    if (cond) {
        console.log('pass:', msg);
    } else {
        console.log('FAIL:', msg);
        fails++;
    }
}
function eq(got, want, msg) {
    ok(got === want, msg + (got === want ? '' : ' — got ' + JSON.stringify(got) + ', want ' + JSON.stringify(want)));
}

// A stand-in for the panel's <pre>: the functions only touch textContent and
// the two properties they hang off it.
function pre(hidePolls) {
    return { textContent: '', _hidePolls: hidePolls };
}

// Lines as vLLM's API server prints them through uvicorn.
const metricsOK = '(APIServer pid=812) INFO:     127.0.0.1:40122 - "GET /metrics HTTP/1.1" 200 OK';
const healthOK = '(APIServer pid=812) INFO:     127.0.0.1:40124 - "GET /health HTTP/1.1" 200 OK';
const healthDown = '(APIServer pid=812) INFO:     127.0.0.1:40126 - "GET /health HTTP/1.1" 503 Service Unavailable';
const chat = '(APIServer pid=812) INFO:     10.0.0.5:51000 - "POST /v1/chat/completions HTTP/1.1" 200 OK';
const models = '(APIServer pid=812) INFO:     10.0.0.5:51002 - "GET /v1/models HTTP/1.1" 200 OK';
const engine = '(EngineCore_DP0 pid=900) INFO 10-06 12:00:01 [gpu_model_runner.py:2007] Model loading took 15.27 GiB';

// ── visibleLog ───────────────────────────────────────────────────────
// The toolchest's own polling is noise; everything else is the log.
{
    const text = [engine, metricsOK, chat, healthOK, models, ''].join('\n');
    eq(visibleLog(pre(true), text), [engine, chat, models, ''].join('\n'),
       'successful /metrics and /health polls are hidden, other requests stay');
    eq(visibleLog(pre(false), text), text,
       'with the filter off the text is returned untouched');
}

// A failing health check is the one poll worth seeing: it is how a wedged
// engine shows itself in the log.
eq(visibleLog(pre(true), healthDown + '\n'), healthDown + '\n',
   'a /health poll that did not return 2xx stays visible');

// Only the exact poll endpoints match. A path that merely starts with one is
// a different request.
{
    const lookalike = '127.0.0.1:1 - "GET /metrics/extra HTTP/1.1" 200 OK';
    eq(visibleLog(pre(true), lookalike), lookalike, '/metrics/extra is not a poll');
}

eq(visibleLog(pre(true), ''), '', 'empty text stays empty');
eq(visibleLog(pre(true), metricsOK + '\n'), '',
   'a chunk that is only a poll line adds nothing, not a blank line');

// ── logPanelAppend ───────────────────────────────────────────────────
// The SSE handler appends one line at a time with its newline.
{
    const p = pre(true);
    [engine, metricsOK, chat, healthOK].forEach(l => logPanelAppend(p, l + '\n'));
    eq(p.textContent, engine + '\n' + chat + '\n', 'appending filters each chunk as it arrives');
    eq(p._rawLog, [engine, metricsOK, chat, healthOK, ''].join('\n'),
       'the unfiltered text is kept, polls included');
}

// The first chunk on page load is the whole buffer at once.
{
    const p = pre(true);
    logPanelAppend(p, [metricsOK, engine, healthOK, chat].join('\n') + '\n');
    eq(p.textContent, engine + '\n' + chat + '\n', 'a multi-line chunk is filtered line by line');
}

// A panel without the poll toggle never sets _hidePolls and shows everything.
{
    const p = { textContent: '' };
    logPanelAppend(p, metricsOK + '\n');
    eq(p.textContent, metricsOK + '\n', 'a panel with no filter shows polls');
}

// ── renderLog ────────────────────────────────────────────────────────
// Turning the filter off must bring hidden lines back in their place, which
// is why the raw text is kept at all.
{
    const p = pre(true);
    [engine, metricsOK, chat].forEach(l => logPanelAppend(p, l + '\n'));
    p._hidePolls = false;
    renderLog(p);
    eq(p.textContent, [engine, metricsOK, chat, ''].join('\n'),
       'turning the filter off restores hidden lines in order');
    p._hidePolls = true;
    renderLog(p);
    eq(p.textContent, engine + '\n' + chat + '\n', 'turning it back on hides them again');
}

// The clear button empties _rawLog; a later render must not resurrect the
// old text, and a pre that never had any renders as empty.
{
    const p = pre(true);
    logPanelAppend(p, engine + '\n');
    p.textContent = '';
    p._rawLog = '';
    renderLog(p);
    eq(p.textContent, '', 'after a clear, render shows nothing');
    const fresh = pre(true);
    renderLog(fresh);
    eq(fresh.textContent, '', 'a pre with no log renders as empty, not "undefined"');
}

if (fails > 0) {
    console.log('\n' + fails + ' failure(s)');
    process.exit(1);
}
console.log('\nall log panel tests passed');

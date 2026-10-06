// Unit tests for followLog in web/static/log-panel.js: a log pane seeded
// with its buffer's backlog and then fed by a live stream.
//
// The order those two arrive in is not under the page's control. The
// backlog must never land on top of lines that streamed in first -- a log
// page once assigned it over them -- so each order is driven here by
// hand, with a stub EventSource and a fetch whose answer the test releases.

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const src = fs.readFileSync(path.join(__dirname, '..', 'static', 'log-panel.js'), 'utf8');

let fails = 0;
function eq(got, want, msg) {
    if (got === want) {
        console.log('pass:', msg);
    } else {
        console.log('FAIL:', msg, '— got', JSON.stringify(got), 'want', JSON.stringify(want));
        fails++;
    }
}

// harness loads log-panel.js with a stream and a backlog the test drives.
function harness() {
    const h = { sources: [], backlog: null };
    class EventSource {
        constructor(url) {
            this.url = url;
            this.listeners = {};
            this.closed = false;
            h.sources.push(this);
        }
        addEventListener(name, fn) { this.listeners[name] = fn; }
        close() { this.closed = true; }
    }
    const fetch = () => new Promise((resolve) => {
        h.backlog = (body) => resolve({
            text: () => Promise.resolve(body),
            json: () => Promise.resolve(JSON.parse(body)),
        });
    });
    const ctx = vm.createContext({ document: { addEventListener() {} }, EventSource, fetch });
    vm.runInContext(src, ctx, { filename: 'log-panel.js' });
    h.followLog = ctx.followLog;
    h.pre = { textContent: '', scrollTop: 0, scrollHeight: 0, closest: () => null };
    h.line = (data) => h.sources[0].onmessage({ data });
    h.settle = () => new Promise((r) => setImmediate(r));
    return h;
}

(async () => {
    {
        const h = harness();
        h.followLog(h.pre, { streamURL: '/s', backlogURL: '/b' });
        h.backlog('old 1\nold 2\n');
        await h.settle();
        h.line('new 1');
        eq(h.pre.textContent, 'old 1\nold 2\nnew 1\n', 'a backlog that arrives first is followed by the stream');
    }
    {
        const h = harness();
        h.followLog(h.pre, { streamURL: '/s', backlogURL: '/b' });
        h.line('new 1');
        h.backlog('old 1\n');
        await h.settle();
        eq(h.pre.textContent, 'new 1\n', 'a backlog that arrives after a live line does not overwrite or precede it');
    }
    {
        const h = harness();
        h.followLog(h.pre, { streamURL: '/s', backlogURL: '/b', backlogJSON: true });
        h.backlog(JSON.stringify(['a', 'b']));
        await h.settle();
        eq(h.pre.textContent, 'a\nb\n', 'a JSON backlog of lines is joined, one per line');
    }
    {
        const h = harness();
        let ended = null;
        h.followLog(h.pre, { streamURL: '/s', backlogURL: '/b', onDone: (e) => { ended = e.data; } });
        h.sources[0].listeners.done({ data: 'job ended' });
        eq(ended, 'job ended', 'the stream ending reaches onDone');
        eq(h.sources[0].closed, true, 'and the stream is closed');
    }

    if (fails > 0) {
        console.log('\n' + fails + ' follow-log test(s) failed');
        process.exit(1);
    }
    console.log('\nall follow-log tests passed');
})();

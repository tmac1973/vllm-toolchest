// Log panel: live-tail auto-scroll + copy to clipboard + clear
document.addEventListener('DOMContentLoaded', () => {
    initLogPanels();
});

document.addEventListener('htmx:afterSettle', () => {
    initLogPanels();
});

function initLogPanels() {
    document.querySelectorAll('.log-panel').forEach(panel => {
        if (panel._logPanelInit) return;
        panel._logPanelInit = true;

        const pre = panel.querySelector('pre');
        const tailToggle = panel.querySelector('.log-tail-toggle');
        const wrapToggle = panel.querySelector('.log-wrap-toggle');
        const pollToggle = panel.querySelector('.log-poll-toggle');
        const copyBtn = panel.querySelector('.log-copy-btn');
        const clearBtn = panel.querySelector('.log-clear-btn');
        if (!pre) return;

        let liveTail = tailToggle ? tailToggle.checked : true;

        if (tailToggle) {
            tailToggle.addEventListener('change', () => {
                liveTail = tailToggle.checked;
                if (liveTail) pre.scrollTop = pre.scrollHeight;
            });
        }

        // Long lines wrap by default. A horizontal scrollbar is not enough:
        // with the overlay scrollbars Linux desktops default to, it is hidden
        // until the pointer finds the box's bottom edge, and a log line
        // reads as cut off. Unwrapped stays available, and is remembered.
        if (wrapToggle) {
            let wrap = true;
            try { wrap = localStorage.getItem('logWrap') !== '0'; } catch (e) {}
            wrapToggle.checked = wrap;
            pre.classList.toggle('log-nowrap', !wrap);
            wrapToggle.addEventListener('change', () => {
                pre.classList.toggle('log-nowrap', !wrapToggle.checked);
                try { localStorage.setItem('logWrap', wrapToggle.checked ? '1' : '0'); } catch (e) {}
                if (liveTail) pre.scrollTop = pre.scrollHeight;
            });
        }

        // The toolchest polls vLLM's /metrics and /health, and uvicorn logs
        // every one of those requests. Hidden by default, and remembered.
        if (pollToggle) {
            let hide = true;
            try { hide = localStorage.getItem('logHidePolls') !== '0'; } catch (e) {}
            pollToggle.checked = hide;
            pre._hidePolls = hide;
            renderLog(pre);
            pollToggle.addEventListener('change', () => {
                pre._hidePolls = pollToggle.checked;
                try { localStorage.setItem('logHidePolls', pollToggle.checked ? '1' : '0'); } catch (e) {}
                renderLog(pre);
            });
        }

        const observer = new MutationObserver(() => {
            if (liveTail) pre.scrollTop = pre.scrollHeight;
        });
        observer.observe(pre, { childList: true, characterData: true, subtree: true });

        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                const text = pre.textContent || pre.innerText;
                copyToClipboard(text).then(() => {
                    const orig = copyBtn.textContent;
                    copyBtn.textContent = 'Copied!';
                    setTimeout(() => { copyBtn.textContent = orig; }, 1500);
                }).catch(() => {
                    copyBtn.textContent = 'Failed';
                    setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
                });
            });
        }

        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                pre.textContent = '';
                pre._rawLog = '';
                // Clearing only the pane leaves the server-side buffer intact,
                // so the next page load brings everything back. A panel that
                // names its buffer gets that emptied too.
                const clearUrl = panel.dataset.clearUrl;
                if (clearUrl) {
                    fetch(clearUrl, { method: 'DELETE' }).catch(() => {});
                }
            });
        }
    });
}

// Only successful polls are hidden: a failing /health stays visible.
const POLL_LINE = /"GET \/(metrics|health) HTTP\/[\d.]+" 2\d\d/;

function visibleLog(pre, text) {
    if (!pre._hidePolls) return text;
    return text.split('\n').filter(line => !POLL_LINE.test(line)).join('\n');
}

// Append log text to a panel's pre. The unfiltered text is kept so that
// turning a filter off brings the hidden lines back.
function logPanelAppend(pre, text) {
    pre._rawLog = (pre._rawLog || '') + text;
    pre.textContent += visibleLog(pre, text);
}

function renderLog(pre) {
    pre.textContent = visibleLog(pre, pre._rawLog || '');
}

// followLog seeds pre with a buffer's backlog, then appends its live stream.
// The backlog is applied only if nothing has streamed in first: applied after
// a live line it would put older lines below newer ones, and assigned over
// the pane it would erase the live ones.
//
// opts.streamURL is the SSE stream; opts.backlogURL answers with the buffer
// as text, or as a JSON array of lines with opts.backlogJSON; opts.onDone
// runs when the stream says it has ended. Returns the EventSource.
function followLog(pre, opts) {
    // A .log-panel scrolls itself, honouring its live-tail toggle; a bare
    // pre is followed always.
    const scroll = () => {
        if (!pre.closest('.log-panel')) pre.scrollTop = pre.scrollHeight;
    };
    const es = new EventSource(opts.streamURL);
    es.onmessage = (e) => {
        logPanelAppend(pre, e.data + '\n');
        scroll();
    };
    es.addEventListener('done', (e) => {
        es.close();
        if (opts.onDone) opts.onDone(e);
    });

    fetch(opts.backlogURL, { headers: { 'HX-Request': 'true' } })
        .then(r => (opts.backlogJSON ? r.json() : r.text()))
        .then(backlog => {
            let text = backlog;
            if (Array.isArray(backlog)) {
                text = backlog.length ? backlog.join('\n') + '\n' : '';
            }
            if (text && !pre._rawLog) {
                logPanelAppend(pre, text);
                pre.scrollTop = pre.scrollHeight;
            }
        })
        .catch(() => {});
    return es;
}

// copyToClipboard falls back to a hidden textarea when the async clipboard
// API is missing or refuses, as it does on plain HTTP to a LAN address -- the
// usual way this is reached.
function copyToClipboard(text) {
    const fallback = () => new Promise((resolve, reject) => {
        try {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.setAttribute('readonly', '');
            ta.style.position = 'fixed';
            ta.style.left = '-9999px';
            document.body.appendChild(ta);
            ta.select();
            const ok = document.execCommand('copy');
            document.body.removeChild(ta);
            if (ok) resolve(); else reject(new Error('copy refused'));
        } catch (e) {
            reject(e);
        }
    });
    if (navigator.clipboard && window.isSecureContext) {
        return navigator.clipboard.writeText(text).catch(fallback);
    }
    return fallback();
}

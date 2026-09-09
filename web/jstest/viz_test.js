// Unit tests for the chart logic in web/static/viz.js.
//
// The interesting parts are decisions about the data — how repeated
// configurations collapse into one heatmap cell, and which charts a given set
// of runs can support — and none of them need a browser. viz.js exports them
// under module.exports when loaded in node, so this runs without a DOM.

const path = require('path');
const viz = require(path.join(__dirname, '..', 'static', 'viz.js'));

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
    ok(JSON.stringify(got) === JSON.stringify(want),
       msg + (JSON.stringify(got) === JSON.stringify(want)
              ? '' : ' — got ' + JSON.stringify(got) + ', want ' + JSON.stringify(want)));
}

// ── grid: several runs can land in one cell ──────────────────────────
// Repeats of the same configuration are the normal case for a job with
// repetitions, and averaging is the only honest thing to draw. Picking the
// first or the best would quietly flatter or penalise a configuration.
const xDim = {name: 'ctx', values: ['8192', '32768'], numeric: true};
const yDim = {name: 'tp', values: ['1', '2'], numeric: true};

const points = [
    {dims: {ctx: '8192', tp: '1'}, metrics: {gen: 100}},
    {dims: {ctx: '8192', tp: '1'}, metrics: {gen: 110}},   // same cell as above
    {dims: {ctx: '32768', tp: '1'}, metrics: {gen: 90}},
    {dims: {ctx: '8192', tp: '2'}, metrics: {gen: 140}}
    // ctx=32768, tp=2 was never run.
];

const g = viz.grid(points, xDim, yDim, 'gen');
eq(g, [[105, 90], [140, null]],
   'repeated configurations average, and a combination never run stays null');

// null rather than 0 matters: zero is a measurement, and a heatmap would
// colour it as the worst result rather than leaving a gap.
ok(g[1][1] === null, 'a missing combination is null, not zero');

// ── chart requirements ───────────────────────────────────────────────
// The page hides controls a chart does not use and refuses one it cannot draw.
// Getting this wrong shows an axis picker that changes nothing, or an empty
// plot with no explanation.
ok(viz.VIZ_CHARTS.scatter.x && !viz.VIZ_CHARTS.scatter.y,
   'scatter takes one dimension');
ok(viz.VIZ_CHARTS.heatmap.x && viz.VIZ_CHARTS.heatmap.y && !viz.VIZ_CHARTS.heatmap.facet,
   'heatmap takes two dimensions');
ok(viz.VIZ_CHARTS.facets.x && viz.VIZ_CHARTS.facets.y && viz.VIZ_CHARTS.facets.facet,
   'faceted heatmaps take three dimensions');
ok(!viz.VIZ_CHARTS.leaderboard.x && viz.VIZ_CHARTS.leaderboard.metric,
   'a leaderboard needs a measurement but no dimension');
ok(!viz.VIZ_CHARTS.pareto.metric,
   'pareto plots two fixed measurements against each other, so it takes no metric picker');

// Every chart the page offers must be described here, or its controls are
// decided by the fallback and quietly wrong.
const offered = ['scatter', 'leaderboard', 'heatmap', 'facets', 'pareto',
                 'parcoords', 'surface3d', 'scatter3d'];
offered.forEach(function(name) {
    ok(Object.prototype.hasOwnProperty.call(viz.VIZ_CHARTS, name),
       'chart "' + name + '" has its requirements declared');
});

if (fails > 0) {
    console.log('\n' + fails + ' failure(s)');
    process.exit(1);
}
console.log('\nall viz tests passed');

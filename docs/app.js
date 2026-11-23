const COLORS = {negative: "#e45756", neutral: "#9d9da1", positive: "#54a24b"};
const PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#72b7b2", "#eeca3b"];
const SENT = ["negative", "neutral", "positive"];
const round = (v, d) => (Math.round(v * 10 ** d + 1e-9) / 10 ** d).toFixed(d);
const pct = v => round(100 * v, 1) + "%";
const f3 = v => round(v, 3);
Chart.defaults.color = "#9aa1b2";
Chart.defaults.borderColor = "#262b38";
Chart.defaults.font.size = 12;
const charts = {};
function draw(id, cfg) { charts[id]?.destroy(); charts[id] = new Chart(document.getElementById(id), cfg); }
const $ = id => document.getElementById(id);
const esc = s => s.replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

// ---------- text processing: mirrors stc/text.py ----------
const ENT = [["&amp;", "&"], ["&lt;", "<"], ["&gt;", ">"], ["&quot;", '"'], ["&#39;", "'"]];
const CONTR = [["can't", "can not"], ["won't", "will not"], ["n't", " not"]];
function clean(t) {
  for (const [k, v] of ENT) t = t.split(k).join(v);
  t = t.toLowerCase().split("\u2019").join("'");
  t = t.replace(/https?:\/\/\S+|www\.\S+/g, " ").replace(/@[\p{L}\p{N}_]+/gu, " ");
  for (const [k, v] of CONTR) t = t.split(k).join(v);
  return t.replace(/\s+/g, " ").trim();
}
const TOKEN_RE = /[a-z0-9']+|[!?]|[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu;
function tokenize(t) { return (t.match(TOKEN_RE) || []).map(x => x.replace(/^'+|'+$/g, "")).filter(Boolean); }

function prepare(m) { m.index = new Map(m.terms.map((t, i) => [t, i])); return m; }
function predict(m, text) {
  const toks = tokenize(clean(text));
  const grams = [...toks];
  for (let i = 0; i + 1 < toks.length; i++) grams.push(toks[i] + " " + toks[i + 1]);
  const tf = new Map();
  for (const g of grams) { const j = m.index.get(g); if (j !== undefined) tf.set(j, (tf.get(j) || 0) + 1); }
  const x = new Map(); let norm = 0;
  for (const [j, c] of tf) { const v = (1 + Math.log(c)) * m.idf[j]; x.set(j, v); norm += v * v; }
  norm = Math.sqrt(norm) || 1;
  for (const [j, v] of x) x.set(j, v / norm);
  const logits = m.classes.map((_, k) => { let s = m.intercept[k]; for (const [j, v] of x) s += v * m.coef[k][j]; return s; });
  const mx = Math.max(...logits), ex = logits.map(l => Math.exp(l - mx)), z = ex.reduce((a, b) => a + b, 0);
  const proba = ex.map(e => e / z);
  const k = proba.indexOf(Math.max(...proba));
  const contrib = [...x].map(([j, v]) => [m.terms[j], v * m.coef[k][j]]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  return {proba, k, contrib};
}

function bars(el, classes, proba, colorFn) {
  const order = classes.map((c, i) => [c, proba[i]]).sort((a, b) => b[1] - a[1]);
  el.innerHTML = order.map(([c, p]) => `<div class="row"><span style="font-size:13px">${esc(c)}</span>
    <div class="track"><div class="fill" style="width:${100 * p}%;background:${colorFn(c)}"></div></div><span>${pct(p)}</span></div>`).join("");
}
function contribChart(id, contrib, k = 10) {
  const top = contrib.slice(0, k);
  draw(id, {type: "bar", data: {labels: top.map(c => c[0]), datasets: [{data: top.map(c => c[1]),
    backgroundColor: top.map(c => c[1] >= 0 ? "#54a24b" : "#e45756")}]},
    options: {indexAxis: "y", plugins: {legend: {display: false}}, scales: {x: {title: {display: true, text: "contribution (log-odds)"}}}}});
}

// ---------- main ----------
Promise.all([fetch("assets/results.json").then(r => r.json()), fetch("assets/model.json").then(r => r.json())]).then(([R, M]) => {
  const S = R.sentiment, names = Object.keys(S), best = R.best_sentiment_model;
  const base = "Logistic Regression (TF-IDF)";
  const ntest = R.y_test_negative.length;
  const cc = R.eda.class_counts, tot = SENT.reduce((a, c) => a + cc[c], 0);
  const K = {
    rows: R.data.clean_rows.toLocaleString(), raw: R.data.raw_rows.toLocaleString(), dups: R.data.dropped_duplicates,
    flagged: R.data.flagged_non_english, ntest: ntest.toLocaleString(),
    bestF1: f3(S[best].macro_f1), baseF1: f3(S[base].macro_f1), topicF1: f3(R.topic[R.best_topic_model].macro_f1),
    balance: SENT.map(c => `${c} ${pct(cc[c] / tot)}`).join(", "),
    lenNeg: R.eda.length_median.negative, lenPos: R.eda.length_median.positive,
  };
  document.querySelectorAll("[data-k]").forEach(el => el.textContent = K[el.dataset.k]);

  // EDA
  draw("cBalance", {type: "bar", data: {labels: SENT, datasets: [{data: SENT.map(c => cc[c]), backgroundColor: SENT.map(c => COLORS[c])}]},
    options: {plugins: {legend: {display: false}}}});
  const bins = R.eda.length_bins;
  draw("cLength", {type: "line", data: {labels: bins.slice(0, -1).map((b, i) => `${b}-${bins[i + 1] - 1}`),
    datasets: SENT.map(c => { const h = R.eda.length_hist[c], s = h.reduce((a, b) => a + b, 0);
      return {label: c, data: h.map(v => v / s), borderColor: COLORS[c], backgroundColor: COLORS[c] + "33", fill: true, tension: .3, pointRadius: 0}; })},
    options: {scales: {y: {title: {display: true, text: "share of class"}}, x: {title: {display: true, text: "tokens"}}}}});
  const tc = Object.entries(R.eda.topic_counts).sort((a, b) => b[1] - a[1]);
  draw("cTopics", {type: "bar", data: {labels: tc.map(t => t[0]), datasets: [{data: tc.map(t => t[1]), backgroundColor: "#e45756"}]},
    options: {indexAxis: "y", plugins: {legend: {display: false}}}});
  const showNgrams = () => { const c = $("ngramSel").value;
    $("ngrams").innerHTML = R.eda.top_bigrams[c].map(([t, n]) => `<span style="background:${COLORS[c]}33">${esc(t)} · ${n}</span>`).join(""); };
  $("ngramSel").onchange = showNgrams; showNgrams();

  // Tables
  const row = (n, cells, cls = "") => `<tr class="${cls}"><td>${n}</td>${cells.map(c => `<td>${c}</td>`).join("")}</tr>`;
  $("tblModels").innerHTML = `<tr><th>Model</th><th>Setup</th><th>Val macro-F1</th><th>Test macro-F1 [95% CI]</th><th>Accuracy</th><th>ROC-AUC</th></tr>` +
    names.map(n => row(n, [`<span class="muted" style="font-size:12px">${esc(S[n].info)}</span>`, S[n].val_macro_f1 == null ? "—" : f3(S[n].val_macro_f1),
      `${f3(S[n].macro_f1)} <span class="muted">[${S[n].macro_f1_ci95.map(f3).join(", ")}]</span>`, pct(S[n].accuracy), f3(S[n].roc_auc_ovr)], n === best ? "best" : "")).join("");
  $("tblRecall").innerHTML = `<tr><th>Model</th>${SENT.map(c => `<th>${c}</th>`).join("")}</tr>` +
    names.map(n => row(n, SENT.map(c => pct(S[n].per_class[c].recall)), n === best ? "best" : "")).join("");
  const T = R.topic, tn = Object.keys(T), topics = R.topic_classes;
  $("tblTopic").innerHTML = `<tr><th>Recall per topic</th>${tn.map(n => `<th>${n}</th>`).join("")}</tr>` +
    topics.map(t => row(`${t} <span class="muted">(n=${T[tn[0]].per_class[t].support})</span>`, tn.map(n => pct(T[n].per_class[t].recall)))).join("") +
    row("<b>Macro-F1</b>", tn.map(n => `<b>${f3(T[n].macro_f1)}</b>`)) + row("Accuracy", tn.map(n => pct(T[n].accuracy)));
  $("tblErrors").innerHTML = `<tr><th>Tweet</th><th>True</th><th>Predicted</th><th>Conf.</th></tr>` +
    R.errors.slice(0, 8).map(e => row(esc(e.text), [e.true, e.pred, pct(e.confidence)])).join("");

  // Narrative bound to numbers
  const ranked = [...names].sort((a, b) => S[b].macro_f1 - S[a].macro_f1);
  const bestT = R.best_topic_model;
  $("resultsText").innerHTML = `<p>On the held-out test set (${ntest.toLocaleString()} tweets) the best sentiment model is <b>${best}</b> with
    <b>macro-F1 ${f3(S[best].macro_f1)}</b> (95% CI ${S[best].macro_f1_ci95.map(f3).join("–")}) and accuracy ${pct(S[best].accuracy)}.
    Ranking: ${ranked.map(n => `${n} ${f3(S[n].macro_f1)}`).join(" › ")}.</p>
    <p>For complaint topics, <b>${bestT}</b> reaches macro-F1 ${f3(T[bestT].macro_f1)} over ${topics.length} classes
    (accuracy ${pct(T[bestT].accuracy)}); large, lexically distinctive topics (cancellations, late flights, lost luggage) are recognised well, while
    vague ones ("Bad Flight", "longlines") are not.</p>`;
  const ft = names.find(n => n.includes("fine-tuned"));
  $("lessons").innerHTML = `<ul class="tight">
    <li>A tuned <b>TF-IDF + logistic regression</b> baseline (macro-F1 ${f3(S[base].macro_f1)}) beats random forest, gradient boosting, frozen embeddings and a zero-shot Twitter model — and it is tiny (runs in this page) and explainable.</li>
    ${ft ? `<li>Fine-tuning a transformer is where contextual models pay off (${f3(S[ft].macro_f1)}, ${S[ft].macro_f1 > S[base].macro_f1 ? "+" : ""}${f3(S[ft].macro_f1 - S[base].macro_f1)} vs baseline); frozen sentence embeddings + LR (${f3(S["MiniLM embeddings + LR"].macro_f1)}) are not enough on their own. The biggest gain is on the hard <b>neutral</b> class (recall ${pct(S[ft].per_class.neutral.recall)} vs ${pct(S[base].per_class.neutral.recall)}).</li>` :
      `<li>Frozen sentence embeddings + LR (${f3(S["MiniLM embeddings + LR"].macro_f1)}) do not beat the sparse baseline on short, slangy tweets.</li>`}
    <li>Tree ensembles on sparse TF-IDF underperform linear models for text (RF ${f3(S["Random Forest (TF-IDF)"].macro_f1)}).</li>
    <li>Preprocessing decisions matter more than expected: removing @handles prevents airline leakage, de-duplicating prevents train/test leakage,
      and language detection must be validated by eye before being used as a filter.</li>
    <li>Macro-F1 + per-class recall exposes the weak <b>neutral</b> class that accuracy hides.</li></ul>`;

  // Comparison chart
  const MK = [["accuracy", "Accuracy"], ["macro_precision", "Precision"], ["macro_recall", "Recall"], ["macro_f1", "F1"], ["roc_auc_ovr", "ROC-AUC"]];
  draw("cCompare", {type: "bar", data: {labels: MK.map(m => m[1]), datasets: names.map((n, i) => ({label: n, data: MK.map(m => S[n][m[0]]), backgroundColor: PALETTE[i]}))},
    options: {scales: {y: {min: 0.4, max: 1, ticks: {callback: v => pct(v)}}}}});

  // 3-class confusion
  const fill = (sel, opts, def) => { $(sel).innerHTML = opts.map(o => `<option ${o === def ? "selected" : ""}>${o}</option>`).join(""); };
  fill("cmModel", names, best); fill("thrModel", names, best);
  const drawCM3 = () => { const cm = S[$("cmModel").value].confusion;
    draw("cCM3", {type: "bar", data: {labels: SENT.map(c => "actual " + c), datasets: SENT.map((c, j) => ({label: "pred " + c, data: cm.map(r => r[j]), backgroundColor: COLORS[c]}))},
      options: {indexAxis: "y", scales: {x: {stacked: true}, y: {stacked: true}}}}); };
  $("cmModel").onchange = drawCM3; drawCM3();

  // Binary threshold
  const yb = R.y_test_negative;
  const drawBin = () => { const t = +$("thr").value, s = S[$("thrModel").value].binary.scores;
    let tp = 0, fp = 0, tn_ = 0, fn = 0;
    s.forEach((v, i) => { const p = v >= t; if (p && yb[i]) tp++; else if (p) fp++; else if (yb[i]) fn++; else tn_++; });
    const prec = tp / (tp + fp || 1), rec = tp / (tp + fn || 1), f1 = 2 * prec * rec / (prec + rec || 1);
    $("thrVal").textContent = t.toFixed(2);
    const cell = (n, l, good) => `<div class="cell" style="background:${good ? "#1f3a24" : "#3a1f22"}"><b>${n}</b>${l}</div>`;
    $("cmBin").innerHTML = `<div></div><div class="muted">Pred. not negative</div><div class="muted">Pred. negative</div>
      <div class="muted" style="align-self:center">Actual not negative</div>${cell(tn_, "True neg.", 1)}${cell(fp, "False pos.", 0)}
      <div class="muted" style="align-self:center">Actual negative</div>${cell(fn, "False neg.", 0)}${cell(tp, "True pos.", 1)}`;
    $("binStats").innerHTML = [["accuracy", (tp + tn_) / s.length], ["precision", prec], ["recall", rec], ["f1", f1]]
      .map(([l, v]) => `<div><b>${pct(v)}</b><span class="muted">${l}</span></div>`).join(""); };
  $("thr").oninput = drawBin; $("thrModel").onchange = drawBin; drawBin();

  // ROC / PR
  const curve = (key, auc, xl, yl) => ({type: "scatter", data: {datasets: names.map((n, i) => ({label: `${n} (${auc} ${f3(S[n].binary[auc === "AUC" ? "roc_auc" : "ap"])})`,
    data: S[n].binary[key].map(([x, y]) => ({x, y})), showLine: true, pointRadius: 0, borderColor: PALETTE[i], borderWidth: 2}))
    .concat(key === "roc" ? [{label: "random", data: [{x: 0, y: 0}, {x: 1, y: 1}], showLine: true, pointRadius: 0, borderColor: "#555", borderDash: [4, 4]}] : [])},
    options: {scales: {x: {min: 0, max: 1, title: {display: true, text: xl}}, y: {min: 0, max: 1, title: {display: true, text: yl}}}, plugins: {legend: {labels: {boxWidth: 10, font: {size: 10}}}}}});
  draw("cROC", curve("roc", "AUC", "False positive rate", "True positive rate"));
  draw("cPR", curve("pr", "AP", "Recall", "Precision"));

  // Importance + coefficients
  draw("cImp", {type: "bar", data: {labels: R.rf_importance.map(r => r[0]), datasets: [{data: R.rf_importance.map(r => r[1]), backgroundColor: "#4c78a8"}]},
    options: {indexAxis: "y", plugins: {legend: {display: false}}}});
  const drawCoef = () => { const c = $("coefSel").value, d = R.lr_top_terms[c];
    draw("cCoef", {type: "bar", data: {labels: d.map(r => r[0]), datasets: [{data: d.map(r => r[1]), backgroundColor: COLORS[c]}]},
      options: {indexAxis: "y", plugins: {legend: {display: false}}, scales: {x: {title: {display: true, text: "coefficient"}}}}}); };
  $("coefSel").onchange = drawCoef; drawCoef();

  // Loss curves
  const lossOpts = [["Gradient boosting (per round)", "gb"]];
  if (R.finetune_history) lossOpts.unshift(["DistilRoBERTa fine-tuning (per epoch)", "ft"]);
  $("lossSel").innerHTML = lossOpts.map(([l, v]) => `<option value="${v}">${l}</option>`).join("");
  const drawLoss = () => { let lab, tr, va, xl;
    if ($("lossSel").value === "gb") { tr = R.gb_loss.train; va = R.gb_loss.val; lab = tr.map((_, i) => i + 1); xl = "Boosting round"; }
    else { const h = R.finetune_history; lab = h.map(e => e.epoch); tr = h.map(e => e.train_loss); va = h.map(e => e.val_loss); xl = "Epoch"; }
    draw("cLoss", {type: "line", data: {labels: lab, datasets: [{label: "Training loss", data: tr, borderColor: "#4c78a8", pointRadius: lab.length > 20 ? 0 : 3},
      {label: "Validation loss", data: va, borderColor: "#f58518", pointRadius: lab.length > 20 ? 0 : 3}]},
      options: {scales: {x: {title: {display: true, text: xl}}, y: {title: {display: true, text: "cross-entropy"}}}}}); };
  $("lossSel").onchange = drawLoss; drawLoss();

  // Explanations
  $("expSel").innerHTML = R.explanations.map((e, i) => `<option value="${i}">Example ${i + 1}: true ${e.true}, predicted ${e.pred}</option>`).join("");
  const drawExp = () => { const e = R.explanations[+$("expSel").value];
    $("expText").innerHTML = `"${esc(e.text)}" — P(neg/neu/pos) = ${e.proba.map(p => p.toFixed(2)).join(" / ")}`;
    contribChart("cExp", e.contributions); };
  $("expSel").onchange = drawExp; drawExp();

  // Live demo
  const sm = prepare(M.sentiment), tm = prepare(M.topic);
  const EX = ["Thanks for the upgrade, the crew was amazing!", "Lost my bag again. Nobody at the desk can tell me anything.",
    "What time does the first flight to Denver leave tomorrow?", "Great, another 3 hour delay. Love it.", "Flight attendant was so rude to my mother"];
  $("examples").innerHTML = EX.map(e => `<button>${esc(e)}</button>`).join("");
  $("examples").querySelectorAll("button").forEach(b => b.onclick = () => { $("inp").value = b.textContent; run(); });
  const run = () => { const t = $("inp").value, s = predict(sm, t);
    bars($("outSent"), sm.classes, s.proba, c => COLORS[c]);
    if (sm.classes[s.k] === "negative") { const tp = predict(tm, t); bars($("outTopic"), tm.classes, tp.proba.map(v => v), () => "#e45756");
      $("outTopic").innerHTML = $("outTopic").innerHTML.split('<div class="row">').slice(0, 5).join('<div class="row">'); }
    else $("outTopic").innerHTML = `<span class="muted">Not predicted negative.</span>`;
    contribChart("cLive", s.contrib.length ? s.contrib : [["(no known n-grams)", 0]]); };
  $("inp").oninput = run; run();
  window.__parity = () => R.explanations.map(e => ({stored: e.proba, js: predict(sm, e.text).proba.map(p => +p.toFixed(3))}));
});

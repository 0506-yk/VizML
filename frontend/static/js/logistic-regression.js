(function () {
  'use strict';

  const $ = id => document.getElementById(id);
  const canvas = $('boundaryCanvas');
  const context = canvas.getContext('2d');
  const state = { result: null, checkpoint: 0, chart: null, timer: null, hover: null };
  const datasetHints = {
    blobs: '适合线性分类器，通常能学到清晰边界。',
    moons: '弯曲结构无法被一条直线完全分开，用来观察欠拟合。',
    circles: '同心圆需要非线性边界，是逻辑回归的经典反例。',
    xor: 'XOR 的对角类别无法被单一直线分开，适合认识模型容量。'
  };

  function setStatus(message, tone) {
    $('status').textContent = message;
    $('status').dataset.state = tone || '';
  }

  function fillRange(input) {
    const width = Number(input.max) - Number(input.min);
    const fill = width ? (Number(input.value) - Number(input.min)) / width * 100 : 0;
    input.style.setProperty('--fill', `${fill}%`);
  }

  [
    ['samples', 'samplesValue', value => value.toFixed(0)],
    ['noise', 'noiseValue', value => value.toFixed(2)],
    ['epochs', 'epochsValue', value => value.toFixed(0)],
    ['learningRate', 'learningRateValue', value => value.toFixed(2)],
    ['regularization', 'regularizationValue', value => value.toFixed(2)],
    ['threshold', 'thresholdValue', value => value.toFixed(2)]
  ].forEach(([inputId, outputId, format]) => {
    const input = $(inputId);
    const update = () => { $(outputId).textContent = format(Number(input.value)); fillRange(input); };
    input.addEventListener('input', update);
    update();
  });

  $('dataset').addEventListener('change', () => {
    $('datasetHint').textContent = datasetHints[$('dataset').value];
  });

  function sigmoid(value) {
    if (value >= 0) return 1 / (1 + Math.exp(-value));
    const exp = Math.exp(value);
    return exp / (1 + exp);
  }

  function currentCheckpoint() {
    return state.result?.checkpoints[state.checkpoint] || null;
  }

  function probability(point, checkpoint = currentCheckpoint()) {
    if (!checkpoint) return .5;
    return sigmoid(checkpoint.coefficients[0] * point[0] + checkpoint.coefficients[1] * point[1] + checkpoint.intercept);
  }

  function metricsAtThreshold() {
    const result = state.result;
    if (!result) return null;
    const threshold = Number($('threshold').value);
    const testSet = new Set(result.data.test_indices);
    let tp = 0, tn = 0, fp = 0, fn = 0;
    const scored = [];
    result.data.X.forEach((point, index) => {
      if (!testSet.has(index)) return;
      const score = probability(point);
      const truth = result.data.y[index];
      const prediction = score >= threshold ? 1 : 0;
      scored.push({ score, truth });
      if (prediction === 1 && truth === 1) tp += 1;
      else if (prediction === 0 && truth === 0) tn += 1;
      else if (prediction === 1) fp += 1;
      else fn += 1;
    });
    const precision = tp / Math.max(1, tp + fp);
    const recall = tp / Math.max(1, tp + fn);
    let concordant = 0, pairs = 0;
    const positives = scored.filter(item => item.truth === 1);
    const negatives = scored.filter(item => item.truth === 0);
    positives.forEach(pos => negatives.forEach(neg => {
      pairs += 1;
      concordant += pos.score > neg.score ? 1 : pos.score === neg.score ? .5 : 0;
    }));
    return {
      tp, tn, fp, fn,
      accuracy: (tp + tn) / Math.max(1, tp + tn + fp + fn),
      precision,
      recall,
      auc: concordant / Math.max(1, pairs)
    };
  }

  function percent(value) { return `${(value * 100).toFixed(1)}%`; }

  function updateDiagnostics() {
    const checkpoint = currentCheckpoint();
    const metrics = metricsAtThreshold();
    if (!checkpoint || !metrics) return;
    $('accuracy').textContent = percent(metrics.accuracy);
    $('precision').textContent = percent(metrics.precision);
    $('recall').textContent = percent(metrics.recall);
    $('auc').textContent = metrics.auc.toFixed(3);
    ['tn', 'fp', 'fn', 'tp'].forEach(key => { $(key).textContent = metrics[key]; });
    const [w1, w2] = checkpoint.coefficients;
    const sign2 = w2 >= 0 ? '+' : '−';
    const signB = checkpoint.intercept >= 0 ? '+' : '−';
    $('modelFormula').textContent = `z = ${w1.toFixed(2)}x₁ ${sign2} ${Math.abs(w2).toFixed(2)}x₂ ${signB} ${Math.abs(checkpoint.intercept).toFixed(2)}`;
    $('lossBadge').textContent = `LOSS ${checkpoint.test_loss.toFixed(3)}`;
    const nonlinear = $('dataset').value !== 'blobs';
    let insight = `当前阈值下，模型正确分类 ${metrics.tp + metrics.tn} / ${metrics.tp + metrics.tn + metrics.fp + metrics.fn} 个测试样本。`;
    if (nonlinear && metrics.accuracy < .85) insight += ' 数据不是线性可分，误差主要来自模型只能画出直线边界。';
    else if (metrics.precision - metrics.recall > .12) insight += ' 精确率明显高于召回率，阈值偏保守。';
    else if (metrics.recall - metrics.precision > .12) insight += ' 召回率明显高于精确率，模型更愿意预测正类。';
    else insight += ' 精确率与召回率较均衡。';
    $('insight').textContent = insight;
  }

  function colorForProbability(p) {
    const low = [232, 234, 251], mid = [249, 250, 253], high = [252, 231, 225];
    const from = p < .5 ? low : mid;
    const to = p < .5 ? mid : high;
    const ratio = p < .5 ? p * 2 : (p - .5) * 2;
    const channels = from.map((value, index) => Math.round(value + (to[index] - value) * ratio));
    return `rgb(${channels.join(',')})`;
  }

  function resizeCanvas() {
    const bounds = canvas.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return null;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(bounds.width * ratio);
    canvas.height = Math.round(bounds.height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { width: bounds.width, height: bounds.height };
  }

  function drawBoundaryLine(bounds, plot, probabilityLevel, style, width, dash) {
    const checkpoint = currentCheckpoint();
    if (!checkpoint) return;
    const [w1, w2] = checkpoint.coefficients;
    const target = Math.log(probabilityLevel / (1 - probabilityLevel)) - checkpoint.intercept;
    const toX = value => plot.left + (value - bounds.x[0]) / (bounds.x[1] - bounds.x[0]) * plot.width;
    const toY = value => plot.top + (bounds.y[1] - value) / (bounds.y[1] - bounds.y[0]) * plot.height;
    context.save(); context.beginPath(); context.rect(plot.left, plot.top, plot.width, plot.height); context.clip(); context.beginPath();
    if (Math.abs(w2) > 1e-9) {
      const firstY = (target - w1 * bounds.x[0]) / w2;
      const lastY = (target - w1 * bounds.x[1]) / w2;
      context.moveTo(toX(bounds.x[0]), toY(firstY)); context.lineTo(toX(bounds.x[1]), toY(lastY));
    } else if (Math.abs(w1) > 1e-9) {
      const x = target / w1; context.moveTo(toX(x), plot.top); context.lineTo(toX(x), plot.top + plot.height);
    }
    context.strokeStyle = style; context.lineWidth = width; context.setLineDash(dash); context.stroke(); context.restore();
  }

  function renderBoundary() {
    const size = resizeCanvas();
    if (!size) return;
    context.clearRect(0, 0, size.width, size.height);
    if (!state.result) return;
    const bounds = state.result.bounds;
    const plot = { left: 44, top: 16, width: size.width - 62, height: size.height - 48 };
    const fromX = pixel => bounds.x[0] + (pixel - plot.left) / plot.width * (bounds.x[1] - bounds.x[0]);
    const fromY = pixel => bounds.y[1] - (pixel - plot.top) / plot.height * (bounds.y[1] - bounds.y[0]);
    const toX = value => plot.left + (value - bounds.x[0]) / (bounds.x[1] - bounds.x[0]) * plot.width;
    const toY = value => plot.top + (bounds.y[1] - value) / (bounds.y[1] - bounds.y[0]) * plot.height;

    const tile = 7;
    for (let y = plot.top; y < plot.top + plot.height; y += tile) {
      for (let x = plot.left; x < plot.left + plot.width; x += tile) {
        context.fillStyle = colorForProbability(probability([fromX(x + tile / 2), fromY(y + tile / 2)]));
        context.fillRect(x, y, tile + .5, tile + .5);
      }
    }
    context.strokeStyle = 'rgba(101,116,139,.12)'; context.lineWidth = 1;
    for (let index = 1; index < 5; index += 1) {
      const x = plot.left + plot.width * index / 5, y = plot.top + plot.height * index / 5;
      context.beginPath(); context.moveTo(x, plot.top); context.lineTo(x, plot.top + plot.height); context.moveTo(plot.left, y); context.lineTo(plot.left + plot.width, y); context.stroke();
    }
    drawBoundaryLine(bounds, plot, .25, 'rgba(80,77,150,.28)', 1, [3, 5]);
    drawBoundaryLine(bounds, plot, .75, 'rgba(190,92,76,.28)', 1, [3, 5]);
    drawBoundaryLine(bounds, plot, Number($('threshold').value), '#293752', 2.2, [8, 5]);

    const testSet = new Set(state.result.data.test_indices);
    state.result.data.X.forEach((point, index) => {
      const isTest = testSet.has(index), label = state.result.data.y[index];
      context.beginPath(); context.arc(toX(point[0]), toY(point[1]), isTest ? 4.7 : 3.7, 0, Math.PI * 2);
      context.fillStyle = label === 0 ? '#5147d9' : '#e56d56'; context.fill();
      context.strokeStyle = isTest ? '#fff' : 'rgba(255,255,255,.8)'; context.lineWidth = isTest ? 2.2 : 1; context.stroke();
      if (isTest) { context.strokeStyle = '#26344d'; context.lineWidth = .8; context.stroke(); }
    });
    context.strokeStyle = '#7d8ba1'; context.lineWidth = 1; context.beginPath(); context.moveTo(plot.left, plot.top); context.lineTo(plot.left, plot.top + plot.height); context.lineTo(plot.left + plot.width, plot.top + plot.height); context.stroke();
    context.fillStyle = '#8a97a9'; context.font = '10px sans-serif'; context.textAlign = 'center';
    context.fillText('特征 x₁', plot.left + plot.width / 2, size.height - 7);
    context.save(); context.translate(12, plot.top + plot.height / 2); context.rotate(-Math.PI / 2); context.fillText('特征 x₂', 0, 0); context.restore();
    if (state.hover && state.hover.x >= plot.left && state.hover.x <= plot.left + plot.width && state.hover.y >= plot.top && state.hover.y <= plot.top + plot.height) {
      const value = probability([fromX(state.hover.x), fromY(state.hover.y)]);
      const label = `P(y=1) ${(value * 100).toFixed(1)}%`;
      context.font = '700 10px sans-serif'; const boxWidth = context.measureText(label).width + 16;
      const x = Math.min(state.hover.x + 10, size.width - boxWidth - 5), y = Math.max(6, state.hover.y - 29);
      context.fillStyle = 'rgba(25,34,55,.88)'; context.beginPath(); context.roundRect(x, y, boxWidth, 23, 6); context.fill();
      context.fillStyle = '#fff'; context.textAlign = 'left'; context.fillText(label, x + 8, y + 15);
    }
  }

  function selectCheckpoint(index) {
    if (!state.result) return;
    state.checkpoint = Math.max(0, Math.min(index, state.result.checkpoints.length - 1));
    $('epochSlider').value = state.checkpoint; fillRange($('epochSlider'));
    const checkpoint = currentCheckpoint();
    $('epochLabel').textContent = `EPOCH ${checkpoint.epoch} / ${state.result.params.epochs}`;
    if (state.chart) state.chart.update('none');
    updateDiagnostics(); renderBoundary();
  }

  const checkpointMarker = { id: 'logisticCheckpoint', afterDraw(chart) {
    const checkpoint = currentCheckpoint();
    if (!checkpoint || !chart.chartArea) return;
    const x = chart.scales.x.getPixelForValue(checkpoint.epoch);
    const ctx = chart.ctx; ctx.save(); ctx.beginPath(); ctx.moveTo(x, chart.chartArea.top); ctx.lineTo(x, chart.chartArea.bottom);
    ctx.strokeStyle = '#5952d2'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]); ctx.stroke(); ctx.restore();
  } };

  function drawLossChart() {
    if (state.chart) state.chart.destroy();
    if (typeof Chart === 'undefined') return;
    const history = state.result.history;
    const chartContext = $('lossChart').getContext('2d');
    state.chart = new Chart(chartContext, { type: 'line', data: { datasets: [
      { label: '训练损失', data: history.map(item => ({ x: item.epoch, y: item.train_loss })), borderColor: '#5952d2', backgroundColor: 'rgba(89,82,210,.08)', borderWidth: 2.2, pointRadius: 0, tension: .18, fill: true },
      { label: '测试损失', data: history.map(item => ({ x: item.epoch, y: item.test_loss })), borderColor: '#e07159', borderWidth: 1.8, pointRadius: 0, tension: .18 }
    ] }, plugins: [checkpointMarker], options: { responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
      interaction: { mode: 'index', intersect: false }, plugins: { legend: { position: 'bottom', align: 'start', labels: { boxWidth: 11, boxHeight: 3, color: '#7e8a9d', font: { size: 10 } } }, tooltip: { callbacks: { title: items => `第 ${items[0].parsed.x} 轮`, label: item => `${item.dataset.label}: ${item.parsed.y.toFixed(4)}` } } },
      scales: { x: { type: 'linear', grid: { display: false }, border: { display: false }, ticks: { color: '#9aa6b6', maxTicksLimit: 6, font: { size: 10 } }, title: { display: true, text: '训练轮数', color: '#9aa6b6', font: { size: 9 } } }, y: { grid: { color: '#eef1f5' }, border: { display: false }, ticks: { color: '#9aa6b6', maxTicksLimit: 5, font: { size: 10 } } } }
    } });
  }

  function stopPlayback() {
    if (state.timer !== null) clearInterval(state.timer);
    state.timer = null;
    $('playButton').innerHTML = '▶ <span>播放学习过程</span>';
  }

  async function train() {
    stopPlayback();
    const button = $('trainButton');
    button.disabled = true; button.querySelector('span').textContent = '正在训练…';
    setStatus('正在生成数据并执行批量梯度下降。');
    try {
      const dataResponse = await fetch('/api/generate_data', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
        type: 'classification', shape: $('dataset').value, n_samples: Number($('samples').value), n_classes: 2,
        noise: Number($('noise').value), random_state: 42, session_namespace: 'logistic_lab'
      }) });
      const data = await dataResponse.json();
      if (!dataResponse.ok || data.status !== 'success') throw new Error(data.message || '数据生成失败');
      const response = await fetch('/api/logistic_regression/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
        session_key: data.session_key, params: { epochs: Number($('epochs').value), learning_rate: Number($('learningRate').value), regularization: Number($('regularization').value), random_state: 42 }
      }) });
      const result = await response.json();
      if (!response.ok || result.status !== 'success') throw new Error(result.message || '模型训练失败');
      state.result = result.results; state.checkpoint = result.results.checkpoints.length - 1;
      $('emptyBoundary').classList.add('is-hidden'); $('emptyChart').classList.add('is-hidden');
      $('threshold').disabled = false; $('epochSlider').disabled = false; $('playButton').disabled = false;
      $('epochSlider').max = result.results.checkpoints.length - 1;
      drawLossChart(); selectCheckpoint(state.checkpoint);
      setStatus(`训练完成 · ${result.results.data.train_indices.length} 个训练样本 · ${result.results.data.test_indices.length} 个测试样本`, 'success');
    } catch (error) {
      setStatus(`实验失败：${error.message}`, 'error');
    } finally {
      button.disabled = false; button.querySelector('span').textContent = state.result ? '重新生成并训练' : '生成数据并训练';
    }
  }

  $('trainButton').addEventListener('click', train);
  $('threshold').addEventListener('input', () => { updateDiagnostics(); renderBoundary(); });
  $('epochSlider').addEventListener('input', event => { stopPlayback(); selectCheckpoint(Number(event.target.value)); });
  $('playButton').addEventListener('click', () => {
    if (!state.result) return;
    if (state.timer !== null) { stopPlayback(); return; }
    let next = state.checkpoint >= state.result.checkpoints.length - 1 ? 0 : state.checkpoint;
    selectCheckpoint(next); $('playButton').innerHTML = 'Ⅱ <span>暂停播放</span>';
    state.timer = setInterval(() => {
      next += 1;
      if (next >= state.result.checkpoints.length) { stopPlayback(); return; }
      selectCheckpoint(next);
    }, 170);
  });
  canvas.addEventListener('pointermove', event => { const rect = canvas.getBoundingClientRect(); state.hover = { x: event.clientX - rect.left, y: event.clientY - rect.top }; renderBoundary(); });
  canvas.addEventListener('pointerleave', () => { state.hover = null; renderBoundary(); });
  if ('ResizeObserver' in window) new ResizeObserver(renderBoundary).observe(canvas.parentElement);
  else window.addEventListener('resize', renderBoundary, { passive: true });
})();

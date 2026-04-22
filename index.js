/**
 * DataLens v2 — Universal Dataset Analyzer & Visualization Engine
 * 
 * A comprehensive analytics toolkit that transforms raw data into
 * actionable insights through statistical analysis, pattern detection,
 * forecasting, and rich ASCII/Unicode visualizations.
 *
 * Key capabilities:
 * - Statistical engine (descriptive, correlations, regression, distributions)
 * - Anomaly detection (IQR + Z-score dual methods)
 * - Simple forecasting via linear extrapolation with confidence bands
 * - Data quality scoring and profiling
 * - 7 visualization types (bar, sparkline, histogram, heatmap, scatter, box, gauge)
 * - Automated actionable insight generation with recommendations
 * - Executive summary with priority ranking
 */

// ─── Math & Stats Library ──────────────────────────────────────────
const sum = a => a.reduce((s, v) => s + v, 0);
const mean = a => a.length ? sum(a) / a.length : 0;
const sortAsc = a => [...a].sort((x, y) => x - y);
const median = a => { const s = sortAsc(a), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m-1]+s[m])/2; };
const variance = a => { const m = mean(a); return mean(a.map(v => (v-m)**2)); };
const stdDev = a => Math.sqrt(variance(a));
const quantile = (a, q) => { const s = sortAsc(a), p = (s.length-1)*q, lo = p|0; return lo===Math.ceil(p) ? s[lo] : s[lo]+(s[Math.ceil(p)]-s[lo])*(p-lo); };
const iqr = a => quantile(a, 0.75) - quantile(a, 0.25);
const skewness = a => { const m = mean(a), sd = stdDev(a), n = a.length; return sd === 0 ? 0 : (n/((n-1)*(Math.max(n-2,1))))*sum(a.map(v=>((v-m)/sd)**3)); };
const kurtosis = a => { const m = mean(a), sd = stdDev(a); return sd === 0 ? 0 : (sum(a.map(v=>((v-m)/sd)**4))/a.length)-3; };
const pearsonR = (x, y) => {
  const mx = mean(x), my = mean(y);
  const num = sum(x.map((v,i) => (v-mx)*(y[i]-my)));
  const den = Math.sqrt(sum(x.map(v=>(v-mx)**2)) * sum(y.map(v=>(v-my)**2)));
  return den === 0 ? 0 : num / den;
};
const linReg = (x, y) => {
  const n = x.length, mx = mean(x), my = mean(y);
  const ssxy = sum(x.map((v,i)=>(v-mx)*(y[i]-my)));
  const ssxx = sum(x.map(v=>(v-mx)**2));
  const slope = ssxx === 0 ? 0 : ssxy / ssxx;
  const intercept = my - slope * mx;
  const pred = x.map(v => slope*v+intercept);
  const ssRes = sum(y.map((v,i)=>(v-pred[i])**2));
  const ssTot = sum(y.map(v=>(v-my)**2));
  const r2 = ssTot === 0 ? 0 : Math.max(0, 1 - ssRes/ssTot);
  const se = n > 2 ? Math.sqrt(ssRes/(n-2)) : 0;
  return { slope, intercept, r2, se, predict: v => slope*v+intercept };
};

// Dual anomaly detection: IQR + Z-score
const detectAnomalies = a => {
  const q1 = quantile(a,0.25), q3 = quantile(a,0.75), iq = q3-q1;
  const lo = q1-1.5*iq, hi = q3+1.5*iq;
  const m = mean(a), sd = stdDev(a);
  return a.map((v,i) => {
    const iqrFlag = v < lo || v > hi;
    const zScore = sd === 0 ? 0 : Math.abs((v-m)/sd);
    const zFlag = zScore > 2.5;
    return (iqrFlag || zFlag) ? { idx: i, value: v, zScore: +zScore.toFixed(2), method: iqrFlag && zFlag ? 'both' : iqrFlag ? 'IQR' : 'z-score' } : null;
  }).filter(Boolean);
};

// Simple forecast via linear extrapolation
const forecast = (values, periods = 3) => {
  const x = values.map((_,i) => i);
  const reg = linReg(x, values);
  const n = values.length;
  const results = [];
  for (let i = 0; i < periods; i++) {
    const xNew = n + i;
    const yHat = reg.predict(xNew);
    results.push({ period: xNew, value: +yHat.toFixed(2), lower: +(yHat - 1.96*reg.se).toFixed(2), upper: +(yHat + 1.96*reg.se).toFixed(2) });
  }
  return { predictions: results, model: reg };
};

// Data quality score
const qualityScore = (data, fields) => {
  let totalChecks = 0, passed = 0;
  const issues = [];
  fields.forEach(f => {
    const vals = data.map(d => d[f]);
    const nulls = vals.filter(v => v == null || isNaN(v)).length;
    totalChecks += 4;
    // Completeness
    if (nulls === 0) passed++; else issues.push(`${f}: ${nulls} missing values`);
    // No negative where unexpected
    const hasNeg = vals.some(v => v < 0);
    if (!hasNeg) passed++; else issues.push(`${f}: contains negative values`);
    // Reasonable variance (not constant)
    if (stdDev(vals.filter(v=>v!=null)) > 0) passed++; else issues.push(`${f}: zero variance (constant)`);
    // Outlier ratio < 10%
    const anomalies = detectAnomalies(vals.filter(v=>v!=null));
    if (anomalies.length / vals.length < 0.1) passed++; else issues.push(`${f}: ${(anomalies.length/vals.length*100).toFixed(0)}% anomalies`);
  });
  return { score: +(passed/totalChecks*100).toFixed(1), passed, total: totalChecks, issues };
};

// ─── Seeded RNG ────────────────────────────────────────────────────
function rng(seed) { let s=seed; return ()=>{ s=(s*16807)%2147483647; return s/2147483647; }; }

// ─── Data Generation ───────────────────────────────────────────────
function genClimate(r) {
  const cities = ['Tokyo','London','New York','Sydney','Mumbai','Cairo','São Paulo','Moscow'];
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const bases = {Tokyo:[5,6,10,15,20,24,28,29,25,19,13,8],London:[5,5,7,9,13,16,18,18,15,12,8,5],
    'New York':[1,2,7,13,18,24,27,26,22,16,10,4],Sydney:[26,26,24,22,19,16,16,17,19,22,24,25],
    Mumbai:[25,26,28,30,32,30,28,28,28,29,28,26],Cairo:[14,15,18,22,27,30,32,32,29,25,20,15],
    'São Paulo':[25,25,24,22,20,18,18,19,20,22,23,24],Moscow:[-6,-5,0,8,15,18,21,19,13,7,0,-5]};
  const data = [];
  cities.forEach(city => {
    const b = bases[city];
    months.forEach((mo,mi) => {
      const temp = +(b[mi] + (r()-0.5)*6).toFixed(1);
      const isMonsoon = city==='Mumbai' && mi>=5 && mi<=8;
      const rain = +Math.max(0, (isMonsoon?250:60) + (r()-0.5)*80).toFixed(1);
      const hum = +Math.min(100, Math.max(20, 50+rain/5+(r()-0.5)*20)).toFixed(1);
      const aqi = +Math.max(10, ({Mumbai:120,Cairo:95,'New York':45,London:35,Tokyo:40,Sydney:25,'São Paulo':55,Moscow:60})[city] + (r()-0.5)*40).toFixed(0);
      data.push({city,month:mo,monthIdx:mi,temp,rainfall:rain,humidity:hum,airQuality:+aqi});
    });
  });
  return {name:'Global Climate Observatory',data,numFields:['temp','rainfall','humidity','airQuality'],catField:'city',timeField:'month'};
}

function genMarket(r) {
  const sectors = ['Tech','Healthcare','Finance','Energy','Retail','Manufacturing'];
  const qs = ['Q1-24','Q2-24','Q3-24','Q4-24','Q1-25','Q2-25','Q3-25','Q4-25'];
  const growth = {Tech:1.04,Healthcare:1.025,Finance:1.01,Energy:0.99,Retail:1.015,Manufacturing:1.005};
  const data = [];
  sectors.forEach(sec => {
    let rev = 50+r()*100, mar = 10+r()*25, emp = 1000+r()*9000;
    qs.forEach((q,qi) => {
      rev *= growth[sec]+(r()-0.5)*0.08;
      mar += (r()-0.5)*3; mar = Math.max(2,Math.min(45,mar));
      emp *= 1+(r()-0.5)*0.04;
      const satisfaction = +Math.min(100,Math.max(30, 65+(mar/3)+(r()-0.5)*15)).toFixed(1);
      data.push({sector:sec,quarter:q,qIdx:qi,revenue:+rev.toFixed(1),margin:+mar.toFixed(1),employees:+emp.toFixed(0),custSatisfaction:satisfaction});
    });
  });
  return {name:'Market Intelligence Dashboard',data,numFields:['revenue','margin','employees','custSatisfaction'],catField:'sector',timeField:'quarter'};
}

function genHealth(r) {
  const regions = ['North America','Europe','East Asia','South Asia','Latin America','Sub-Saharan Africa','Middle East','Oceania'];
  const data = [];
  const base = {
    'North America':[78,10000,36,90],'Europe':[80,5000,23,92],'East Asia':[79,3500,6,95],
    'South Asia':[69,200,5,75],'Latin America':[74,1000,24,80],'Sub-Saharan Africa':[63,150,11,55],
    'Middle East':[73,1500,30,82],'Oceania':[82,5500,30,93]
  };
  regions.forEach(reg => {
    const [le,hs,ob,vc] = base[reg];
    data.push({
      region: reg,
      lifeExpectancy: +(le+(r()-0.5)*4).toFixed(1),
      healthSpend: +(hs*(0.8+r()*0.4)).toFixed(0),
      obesityRate: +(ob+(r()-0.5)*8).toFixed(1),
      vaccRate: +Math.min(99,Math.max(30,vc+(r()-0.5)*10)).toFixed(1),
      infantMortality: +(({
        'North America':5,'Europe':4,'East Asia':6,'South Asia':30,'Latin America':15,
        'Sub-Saharan Africa':50,'Middle East':18,'Oceania':3
      })[reg]*(0.8+r()*0.4)).toFixed(1),
      physiciansPer1000: +(({
        'North America':2.6,'Europe':3.5,'East Asia':2.0,'South Asia':0.8,'Latin America':1.8,
        'Sub-Saharan Africa':0.2,'Middle East':1.5,'Oceania':3.5
      })[reg]*(0.8+r()*0.4)).toFixed(1)
    });
  });
  return {name:'Global Health Equity Index',data,numFields:['lifeExpectancy','healthSpend','obesityRate','vaccRate','infantMortality','physiciansPer1000'],catField:'region'};
}

// ─── Visualization Engine ──────────────────────────────────────────
const BLOCKS = '▏▎▍▌▋▊▉█', SPARKS = '▁▂▃▄▅▆▇█';

function vizBar(labels, vals, {title='',w=38,unit=''}={}) {
  const mx = Math.max(...vals.map(Math.abs));
  const mxL = Math.max(...labels.map(l => l.length), 6);
  let o = title ? `\n  ${title}\n` : '';
  vals.forEach((v,i) => {
    const len = mx===0 ? 0 : Math.round((Math.abs(v)/mx)*w);
    const bar = v >= 0 ? '█'.repeat(len) : '░'.repeat(len);
    const sign = v < 0 ? '-' : ' ';
    o += `  ${labels[i].padStart(mxL)} │${sign}${bar} ${v.toFixed(1)}${unit}\n`;
  });
  return o;
}

function vizSpark(vals, {label=''}={}) {
  const mn = Math.min(...vals), mx = Math.max(...vals), rng = mx-mn||1;
  const s = vals.map(v => SPARKS[Math.min(7,((v-mn)/rng*7.99)|0)]).join('');
  const trend = vals[vals.length-1] > vals[0] ? '↗' : vals[vals.length-1] < vals[0] ? '↘' : '→';
  return `  ${label?label+' ':''}${s} ${trend} (${mn.toFixed(1)}…${mx.toFixed(1)})`;
}

function vizHist(vals, {title='',bins=10,w=30}={}) {
  const mn = Math.min(...vals), mx = Math.max(...vals), bw = (mx-mn)/bins||1;
  const counts = Array(bins).fill(0);
  vals.forEach(v => { counts[Math.min(bins-1,((v-mn)/bw)|0)]++; });
  const mxC = Math.max(...counts);
  let o = title ? `\n  ${title}\n` : '';
  counts.forEach((c,i) => {
    const lo = (mn+i*bw).toFixed(1), hi = (mn+(i+1)*bw).toFixed(1);
    const bar = mxC===0 ? '' : '█'.repeat(Math.round(c/mxC*w));
    o += `  ${lo.padStart(8)}–${hi.padEnd(8)}│${bar} ${c}\n`;
  });
  return o;
}

function vizHeatmap(matrix, rows, cols, {title=''}={}) {
  const shades = [' ','░','▒','▓','█'];
  const flat = matrix.flat().filter(v=>v!=null);
  const mn = Math.min(...flat), mx = Math.max(...flat), rng = mx-mn||1;
  const mxR = Math.max(...rows.map(r=>r.length),6);
  let o = title ? `\n  ${title}\n` : '';
  o += ' '.repeat(mxR+3) + cols.map(c=>c.slice(0,4).padEnd(5)).join('') + '\n';
  matrix.forEach((row,i) => {
    o += `  ${rows[i].padStart(mxR)} │`;
    row.forEach(v => { const idx = Math.min(4,((v-mn)/rng*4.99)|0); o += ` ${shades[idx]}${shades[idx]}${shades[idx]} `; });
    o += '\n';
  });
  o += `  Scale: ${shades.map((s,i)=>`${s}${s}=${(mn+i/4*rng).toFixed(0)}`).join('  ')}\n`;
  return o;
}

function vizScatter(xv, yv, {title='',w=48,h=16,xLbl='x',yLbl='y'}={}) {
  const xMn=Math.min(...xv),xMx=Math.max(...xv),yMn=Math.min(...yv),yMx=Math.max(...yv);
  const xR=xMx-xMn||1,yR=yMx-yMn||1;
  const grid = Array.from({length:h},()=>Array(w).fill(' '));
  // Add trend line
  const reg = linReg(xv, yv);
  for(let c=0;c<w;c++) {
    const x = xMn + (c/w)*xR;
    const y = reg.predict(x);
    const row = h-1-Math.round(((y-yMn)/yR)*(h-1));
    if(row>=0 && row<h) grid[row][c] = '·';
  }
  xv.forEach((x,i) => {
    const c = Math.min(w-1,((x-xMn)/xR*(w-1))|0);
    const row = Math.min(h-1,h-1-(((yv[i]-yMn)/yR*(h-1))|0));
    grid[row][c] = '●';
  });
  let o = title ? `\n  ${title}\n` : '';
  o += `  ${yLbl}\n`;
  grid.forEach((row,ri) => {
    const lbl = ri===0?yMx.toFixed(0).padStart(7):ri===h-1?yMn.toFixed(0).padStart(7):'       ';
    o += `  ${lbl}│${row.join('')}│\n`;
  });
  o += `        └${'─'.repeat(w)}┘\n`;
  o += `         ${xLbl}: ${xMn.toFixed(0)}${' '.repeat(Math.max(0,w-String(xMn.toFixed(0)).length-String(xMx.toFixed(0)).length))}${xMx.toFixed(0)}\n`;
  o += `         (trend: y = ${reg.slope.toFixed(3)}x + ${reg.intercept.toFixed(1)}, R²=${reg.r2.toFixed(3)})\n`;
  return o;
}

function vizBoxPlot(fields, data, {title=''}={}) {
  const w = 40;
  let o = title ? `\n  ${title}\n` : '';
  fields.forEach(f => {
    const vals = data.map(d=>d[f]).filter(v=>v!=null);
    const mn=Math.min(...vals),mx=Math.max(...vals),q1=quantile(vals,0.25),med=median(vals),q3=quantile(vals,0.75),rng=mx-mn||1;
    const pos = v => Math.round(((v-mn)/rng)*(w-1));
    const line = Array(w).fill('─');
    line[pos(q1)] = '┤'; line[pos(q3)] = '├';
    for(let i=pos(q1)+1;i<pos(q3);i++) line[i]='█';
    line[pos(med)] = '│';
    line[0] = '◄'; line[w-1] = '►';
    o += `  ${f.padStart(18)} ${line.join('')}\n`;
    o += `  ${''.padStart(18)} ${mn.toFixed(1).padStart(6)}${' '.repeat(Math.max(0,w-18))}${mx.toFixed(1)}\n`;
  });
  return o;
}

function vizGauge(label, value, max, {w=30,thresholds=[60,80]}={}) {
  const pct = Math.min(100, (value/max)*100);
  const filled = Math.round(pct/100*w);
  const icon = pct >= thresholds[1] ? '🟢' : pct >= thresholds[0] ? '🟡' : '🔴';
  return `  ${icon} ${label.padEnd(20)} [${'█'.repeat(filled)}${'░'.repeat(w-filled)}] ${pct.toFixed(0)}%`;
}

function vizCorrMatrix(fields, data) {
  const vals = fields.map(f => data.map(d=>d[f]));
  const matrix = fields.map((_,i)=>fields.map((_,j)=>pearsonR(vals[i],vals[j])));
  const mxL = Math.max(...fields.map(f=>f.length),6);
  let o = '\n  Correlation Matrix (Pearson r)\n';
  o += ' '.repeat(mxL+3)+fields.map(f=>f.slice(0,8).padEnd(9)).join('')+'\n';
  const sym = v => v>0.7?'██':v>0.3?'▓▓':v>-0.3?'░░':v>-0.7?'▒▒':'  ';
  matrix.forEach((row,i) => {
    o += `  ${fields[i].padStart(mxL)} │`+row.map(v=>` ${v>=0?' ':''}${v.toFixed(2)}${sym(v)}`).join('')+'\n';
  });
  o += `  Key: ██strong+ ▓▓mod+ ░░weak ▒▒mod- (empty)strong-\n`;
  return o;
}

// ─── Insight Engine ────────────────────────────────────────────────
function generateInsights(ds) {
  const {data, numFields, catField} = ds;
  const insights = [];
  const add = (type, sev, msg, rec) => insights.push({type,severity:sev,message:msg,recommendation:rec});

  numFields.forEach(f => {
    const vals = data.map(d=>d[f]).filter(v=>v!=null);
    const anom = detectAnomalies(vals);
    const sk = skewness(vals), ku = kurtosis(vals);
    if (anom.length > 0) {
      const pct = (anom.length/vals.length*100).toFixed(0);
      add('anomaly','high',`${f}: ${anom.length} anomalies (${pct}%) detected via dual IQR+Z-score`,
        `Investigate extreme values; consider robust measures (median, trimmed mean) for this field`);
    }
    if (Math.abs(sk)>1) add('distribution','medium',`${f}: ${sk>0?'right':'left'}-skewed (skew=${sk.toFixed(2)})`,
      `Apply ${sk>0?'log':'square'} transformation for normalization before modeling`);
    if (ku>2) add('distribution','low',`${f}: heavy tails (kurtosis=${ku.toFixed(2)})`,
      `Use non-parametric methods; median/IQR more reliable than mean/stddev`);
  });

  // Cross-field correlations
  for (let i=0;i<numFields.length;i++) for (let j=i+1;j<numFields.length;j++) {
    const r = pearsonR(data.map(d=>d[numFields[i]]),data.map(d=>d[numFields[j]]));
    if (Math.abs(r)>0.6) add('correlation','high',
      `${r>0?'Positive':'Negative'} correlation (r=${r.toFixed(3)}) between ${numFields[i]} and ${numFields[j]}`,
      `${Math.abs(r)>0.85?'Near-linear relationship — potential predictor':'Moderate association — explore causal mechanisms'}`);
  }

  // Category disparities
  if (catField) {
    const cats = [...new Set(data.map(d=>d[catField]))];
    numFields.forEach(f => {
      const avgs = cats.map(c=>({cat:c,avg:mean(data.filter(d=>d[catField]===c).map(d=>d[f]))})).sort((a,b)=>b.avg-a.avg);
      if (avgs.length>=2) {
        const ratio = Math.abs(avgs[0].avg) / (Math.abs(avgs[avgs.length-1].avg)||1);
        if (ratio>2) add('disparity','high',
          `${f}: ${ratio.toFixed(1)}× gap — ${avgs[0].cat} (${avgs[0].avg.toFixed(1)}) vs ${avgs[avgs.length-1].cat} (${avgs[avgs.length-1].avg.toFixed(1)})`,
          `Target interventions at ${avgs[avgs.length-1].cat}; investigate structural causes of disparity`);
      }
    });
  }

  return insights.sort((a,b)=>{const o={high:0,medium:1,low:2};return o[a.severity]-o[b.severity];});
}

// ─── Report Builder ────────────────────────────────────────────────
function buildReport(ds) {
  const {data,name,numFields,catField,timeField} = ds;
  const cats = catField ? [...new Set(data.map(d=>d[catField]))] : [];
  const times = timeField ? [...new Set(data.map(d=>d[timeField]))] : [];
  let o = '';

  o += '\n╔'+('═'.repeat(68))+'╗\n';
  o += '║  📊 '+name.padEnd(63)+'║\n';
  o += '╠'+('═'.repeat(68))+'╣\n';
  o += `║  Records: ${String(data.length).padEnd(6)} Fields: ${String(numFields.length).padEnd(3)} numeric`;
  if (catField) o += ` │ Groups: ${String(cats.length).padEnd(3)}`;
  if (timeField) o += ` │ Periods: ${times.length}`;
  o += ' '.repeat(Math.max(0,68-49))+' ║\n';
  o += '╚'+('═'.repeat(68))+'╝\n';

  // Data Quality
  const dq = qualityScore(data, numFields);
  o += '\n  🔍 DATA QUALITY PROFILE\n  ' + '─'.repeat(64) + '\n';
  o += vizGauge('Overall Quality', dq.score, 100, {thresholds:[70,90]}) + '\n';
  o += `  Checks: ${dq.passed}/${dq.total} passed\n`;
  if (dq.issues.length) o += `  Issues: ${dq.issues.join('; ')}\n`;

  // Stats Table
  o += '\n  📈 STATISTICAL SUMMARY\n  ' + '─'.repeat(64) + '\n';
  const cw = 13;
  o += '  ' + 'Stat'.padEnd(11) + numFields.map(f=>f.slice(0,cw-1).padStart(cw)).join('') + '\n';
  o += '  ' + '─'.repeat(11+numFields.length*cw) + '\n';
  const allStats = {};
  numFields.forEach(f => {
    const v = data.map(d=>d[f]).filter(x=>x!=null);
    allStats[f] = {mean:mean(v),median:median(v),stdDev:stdDev(v),min:Math.min(...v),max:Math.max(...v),
      q1:quantile(v,0.25),q3:quantile(v,0.75),skewness:skewness(v),kurtosis:kurtosis(v),cv:stdDev(v)/mean(v)*100};
  });
  ['mean','median','stdDev','min','max','q1','q3','skewness','cv'].forEach(m => {
    o += '  ' + (m==='cv'?'CV%':m).padEnd(11);
    numFields.forEach(f => o += String(allStats[f][m].toFixed(2)).padStart(cw));
    o += '\n';
  });

  // Box Plots
  o += '\n  📦 DISTRIBUTION BOX PLOTS\n  ' + '─'.repeat(64) + '\n';
  o += vizBoxPlot(numFields, data, {title:''});

  // Sparklines
  if (catField && data.length > 5) {
    o += '\n  📉 TREND SPARKLINES\n  ' + '─'.repeat(64) + '\n';
    numFields.slice(0,3).forEach(f => {
      o += `\n  ${f}:\n`;
      cats.slice(0,8).forEach(c => {
        const vals = data.filter(d=>d[catField]===c).map(d=>d[f]).filter(v=>v!=null);
        if (vals.length > 2) o += vizSpark(vals, {label:c.padEnd(18).slice(0,18)}) + '\n';
      });
    });
  }

  // Bar Charts
  if (catField) {
    o += '\n  📊 CATEGORY RANKINGS\n  ' + '─'.repeat(64) + '\n';
    numFields.slice(0,2).forEach(f => {
      const avgs = cats.map(c => mean(data.filter(d=>d[catField]===c).map(d=>d[f])));
      o += vizBar(cats, avgs, {title:`Mean ${f} by ${catField}`});
    });
  }

  // Histogram
  o += '\n  📊 DISTRIBUTIONS\n  ' + '─'.repeat(64) + '\n';
  numFields.slice(0,2).forEach(f => {
    o += vizHist(data.map(d=>d[f]).filter(v=>v!=null), {title:f,bins:10});
  });

  // Correlation
  if (numFields.length >= 2) {
    o += '\n  🔗 CORRELATIONS\n  ' + '─'.repeat(64) + '\n';
    o += vizCorrMatrix(numFields, data);

    // Best scatter
    let best = 0, bp = [0,1];
    for (let i=0;i<numFields.length;i++) for(let j=i+1;j<numFields.length;j++) {
      const c = Math.abs(pearsonR(data.map(d=>d[numFields[i]]),data.map(d=>d[numFields[j]])));
      if (c>best) { best=c; bp=[i,j]; }
    }
    const f1=numFields[bp[0]],f2=numFields[bp[1]];
    o += vizScatter(data.map(d=>d[f1]),data.map(d=>d[f2]),{title:`Strongest pair: ${f1} vs ${f2}`,xLbl:f1,yLbl:f2,w:44,h:14});
  }

  // Heatmap
  if (catField && timeField) {
    o += '\n  🗺️  HEATMAP\n  ' + '─'.repeat(64) + '\n';
    const f = numFields[0];
    const matrix = cats.map(c=>times.map(t=>{const m=data.find(d=>d[catField]===c&&d[timeField]===t);return m?m[f]:0;}));
    o += vizHeatmap(matrix, cats, times, {title:`${f} by ${catField} × ${timeField}`});
  }

  // Forecast
  if (numFields.length > 0 && data.length > 6) {
    o += '\n  🔮 FORECAST (Linear Extrapolation, 95% CI)\n  ' + '─'.repeat(64) + '\n';
    numFields.slice(0,3).forEach(f => {
      const vals = data.map(d=>d[f]).filter(v=>v!=null);
      const fc = forecast(vals, 3);
      o += `  ${f}: `;
      fc.predictions.forEach((p,i) => o += `T+${i+1}: ${p.value} [${p.lower}–${p.upper}]  `);
      o += `(R²=${fc.model.r2.toFixed(3)})\n`;
    });
  }

  // Insights
  const insights = generateInsights(ds);
  if (insights.length) {
    o += '\n  💡 ACTIONABLE INSIGHTS\n  ' + '─'.repeat(64) + '\n';
    const ic = {high:'🔴',medium:'🟡',low:'🟢'};
    insights.forEach(ins => {
      o += `  ${ic[ins.severity]} ${ins.message}\n`;
      o += `     ➜ ${ins.recommendation}\n`;
    });
  }

  o += '\n' + '═'.repeat(70) + '\n';
  return o;
}

// ─── Main ──────────────────────────────────────────────────────────
function main() {
  const r = rng(42);
  const datasets = [genClimate(r), genMarket(r), genHealth(r)];

  console.log('\n╔══════════════════════════════════════════════════════════════════════╗');
  console.log('║  🔬 DataLens v2 — Universal Dataset Analyzer & Visualization Engine ║');
  console.log('║  Statistical Analysis • Pattern Detection • Forecasting • Insights  ║');
  console.log('╚══════════════════════════════════════════════════════════════════════╝');

  datasets.forEach(ds => console.log(buildReport(ds)));

  // Executive Summary
  console.log('\n╔══════════════════════════════════════════════════════════════════════╗');
  console.log('║  🌐 EXECUTIVE SUMMARY — Cross-Dataset Intelligence                  ║');
  console.log('╚══════════════════════════════════════════════════════════════════════╝');

  let totalIns = 0, totalRec = 0, allHigh = [];
  datasets.forEach(ds => {
    const ins = generateInsights(ds);
    totalIns += ins.length; totalRec += ds.data.length;
    const high = ins.filter(i=>i.severity==='high');
    allHigh.push(...high.map(h=>({dataset:ds.name,...h})));
    const dq = qualityScore(ds.data, ds.numFields);
    console.log(`\n  📊 ${ds.name}`);
    console.log(`     Records: ${ds.data.length} │ Quality: ${dq.score}% │ Insights: ${ins.length} (${high.length} critical)`);
  });

  console.log(`\n  ━━━━ TOP FINDINGS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
  allHigh.slice(0,8).forEach(h => {
    console.log(`  🔴 [${h.dataset.split(' ')[0]}] ${h.message}`);
    console.log(`     ➜ ${h.recommendation}`);
  });

  console.log(`\n  ━━━━ METHODOLOGY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
  console.log(`  ${totalRec} records across ${datasets.length} datasets`);
  console.log(`  ${totalIns} insights (${allHigh.length} high-priority)`);
  console.log(`  Methods: Descriptive Stats, Pearson Correlation, OLS Regression,`);
  console.log(`  Dual Anomaly Detection (IQR+Z-score), Distribution Analysis,`);
  console.log(`  Linear Forecasting with 95% CI, Data Quality Profiling`);
  console.log(`\n${'═'.repeat(70)}\n`);
  
  return { totalRecords: totalRec, totalInsights: totalIns, datasets: datasets.length };
}

// ─── Exports ───────────────────────────────────────────────────────
const result = main();

// Module exports for sandbox detection
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    analyze: buildReport,
    generateInsights,
    forecast,
    qualityScore,
    detectAnomalies,
    vizBar, vizSpark, vizHist, vizHeatmap, vizScatter, vizBoxPlot, vizCorrMatrix, vizGauge,
    stats: { mean, median, stdDev, variance, skewness, kurtosis, pearsonR, linReg, quantile },
    result
  };
}

// Named exports for ES modules
if (typeof exports !== 'undefined') {
  exports.analyze = buildReport;
  exports.generateInsights = generateInsights;
  exports.forecast = forecast;
  exports.qualityScore = qualityScore;
  exports.detectAnomalies = detectAnomalies;
  exports.result = result;
}

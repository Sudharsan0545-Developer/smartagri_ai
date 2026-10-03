const tabs = document.querySelectorAll('.tab');
const modules = document.querySelectorAll('.module');
tabs.forEach(t => t.addEventListener('click', () => {
  tabs.forEach(x => x.classList.remove('active'));
  modules.forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById(t.dataset.tab).classList.add('active');
  window.scrollTo({top: document.querySelector('.module.active').offsetTop - 90, behavior: 'smooth'});
}));

function formBody(form){ return new URLSearchParams(new FormData(form)); }
async function post(url, form){
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:formBody(form)});
  let data; try { data = await r.json(); } catch { throw new Error('The server returned an unexpected response.'); }
  if(!r.ok || data.ok === false) throw new Error(data.error || 'Something went wrong.');
  return data;
}
function money(n){ return Number(n).toLocaleString('en-IN',{maximumFractionDigits:2}); }
function esc(s){ return String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }
function cropEmoji(crop){
  const n=String(crop).toLowerCase();
  if(n.includes('maize')) return '🌽'; if(n.includes('rice')) return '🌾'; if(n.includes('wheat')) return '🌾';
  if(n.includes('cotton')) return '🌸'; if(n.includes('groundnut')) return '🥜'; if(n.includes('sugar')) return '🎋';
  if(n.includes('coffee')) return '☕'; if(n.includes('turmeric')) return '🟡'; if(n.includes('tea')) return '🍃';
  if(n.includes('banana')) return '🍌'; if(n.includes('chickpea')) return '🫘'; if(n.includes('lentil')) return '🫘';
  return '🌱';
}
function bars(items){
  return `<div class="input-bars">${items.map(x=>`<div class="bar-row"><div class="bar-label"><span>${x.icon||'🌱'} ${esc(x.label)}</span><b>${esc(x.value)}</b></div><div class="bar-track"><i style="width:${Math.max(4,Math.min(100,Number(x.score)||0))}%"></i></div></div>`).join('')}</div>`;
}
function modelPanel(title, items, isPrice=false){
  return `<div class="model-panel"><div class="model-title"><span>🤖</span><div><b>${esc(title)}</b><small>Independent model prediction</small></div></div>${items.map((x,i)=>`<div class="prediction-line"><span class="rank-num">${i+1}</span><span class="pred-crop">${cropEmoji(x.crop)} ${esc(x.crop)}</span><div class="mini-track"><i style="width:${Math.min(100,Math.max(2,x.confidence||0))}%"></i></div><strong>${isPrice?'₹'+money(x.value):Number(x.confidence||0).toFixed(1)+'%'}</strong></div>`).join('')}</div>`;
}
function modelRegressionPanel(items){
  return `<div class="model-grid">${items.slice(0,2).map((x,i)=>`<div class="regression-card ${i===0?'highlight':''}"><span>${i===0?'🌲':'🌳'}</span><div><small>Model ${i+1}</small><b>${esc(x.name)}</b><strong>${Number(x.value).toLocaleString(undefined,{maximumFractionDigits:3})}</strong></div></div>`).join('')}</div>`;
}
function regressionBars(items, currency=false){
  const max=Math.max(...items.map(x=>Number(x.value)),1);
  return `<div class="compare-chart">${items.map(x=>`<div class="compare-row"><span>${esc(x.name)}</span><div><i style="width:${Math.max(7,Number(x.value)/max*100)}%"></i></div><b>${currency?'₹'+money(x.value):Number(x.value).toFixed(3)}</b></div>`).join('')}</div>`;
}
function lineChart(points){
  if(!points || points.length<2) return '';
  const w=780,h=230,p=26; const vals=points.map(x=>Number(x.price)); const min=Math.min(...vals), max=Math.max(...vals), span=max-min||1;
  const coords=points.map((x,i)=>{const px=p+i*(w-2*p)/(points.length-1);const py=h-p-(Number(x.price)-min)/span*(h-2*p);return [px,py];});
  const poly=coords.map(a=>a.join(',')).join(' ');
  const dots=coords.map(a=>`<circle cx="${a[0]}" cy="${a[1]}" r="4"/>`).join('');
  return `<div class="svg-chart"><svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline points="${poly}"/><g>${dots}</g></svg><div class="chart-foot"><span>${esc(points[0].month)}</span><span>${esc(points[Math.floor(points.length/2)].month)}</span><span>${esc(points[points.length-1].month)}</span></div></div>`;
}
function setLoading(box,msg){ box.classList.remove('hidden'); box.innerHTML=`<div class="loading-card"><div class="spinner">🌱</div><b>${msg}</b><span>Please use the result as an advisory, not a guarantee.</span></div>`; }

const stateSelect=document.getElementById('stateSelect');
const districtSelect=document.getElementById('districtSelect');
function updateDistricts(){
  if(!stateSelect||!districtSelect||!window.FARMER_OPTIONS) return;
  districtSelect.innerHTML='';
  (window.FARMER_OPTIONS.districts?.[stateSelect.value]||[]).forEach((d,i)=>{const o=document.createElement('option');o.value=d;o.textContent=d;if(i===0)o.selected=true;districtSelect.appendChild(o);});
}
if(stateSelect){stateSelect.addEventListener('change',updateDistricts);updateDistricts();}

const cropForm=document.getElementById('cropForm');
if(cropForm) cropForm.addEventListener('submit',async e=>{
  e.preventDefault(); const box=document.getElementById('cropResult'); setLoading(box,'Analysing your field conditions…');
  try{
    const d=await post('/api/crop-recommend',cropForm);
    const combined=d.combined, best=d.best;
    const weather=d.weather_model.map(x=>({...x,value:x.confidence})); const local=d.local_model.map(x=>({...x,value:x.confidence}));
    box.innerHTML=`<div class="result-hero"><div class="result-crop-icon">${cropEmoji(best.crop)}</div><div><small>Combined recommendation</small><h2>${esc(best.crop)}</h2><p>Best overall fit from the farmer-condition model + local historical model.</p></div><div class="score-ring"><b>${best.confidence}%</b><small>AI score</small></div></div>
      <div class="dashboard-grid two"><div>${modelPanel('Soil & Weather Random Forest',weather)}</div><div>${modelPanel('Local History Extra Trees',local)}</div></div>
      <div class="dashboard-grid two"><div class="analysis-card"><div class="card-heading"><span>📊</span><div><b>Your input analysis</b><small>Higher bars mean the selected condition is more favourable for modelling.</small></div></div>${bars(d.analysis)}</div>
      <div class="analysis-card"><div class="card-heading"><span>🧭</span><div><b>What the result means</b><small>Crop profile comparison for the recommended crop.</small></div></div><div class="advice-list">${d.advice.map(x=>`<div>✅ ${esc(x)}</div>`).join('')}</div><div class="profile-chip">Typical profile: ${esc(d.profile?.Soil_Fertility||'—')} soil • ${esc(d.profile?.Water_Availability||'—')} water • ${esc(d.profile?.Rainfall_Level||'—')} rain • ${esc(d.profile?.Temperature_Level||'—')} temperature</div></div></div>
      <div class="analysis-card"><div class="card-heading"><span>📍</span><div><b>Local historical crops</b><small>${esc(d.location.State_Name)} • ${esc(d.location.District_Name)} • ${esc(d.location.Season)}</small></div></div>${d.history_exact?.length?`<div class="history-grid">${d.history_exact.slice(0,4).map((x,i)=>`<div class="history-item"><b>${i+1}. ${cropEmoji(x.crop)} ${esc(x.crop)}</b><span>Yield profile ${x.mean_yield}</span><span>Area share ${x.area_share}%</span></div>`).join('')}</div>`:`<p class="muted">No exact local historical combination was found, so the local AI model used similar training patterns.</p>`}</div>`;
  }catch(err){box.innerHTML=`<div class="error-card">⚠️ ${esc(err.message)}</div>`;}
});

const yieldForm=document.getElementById('yieldForm');
if(yieldForm) yieldForm.addEventListener('submit',async e=>{
  e.preventDefault(); const box=document.getElementById('yieldResult'); setLoading(box,'Estimating your harvest…');
  try{
    const d=await post('/api/yield-predict',yieldForm);
    box.innerHTML=`<div class="result-hero"><div class="result-crop-icon">${cropEmoji(yieldForm.elements.Crop.value)}</div><div><small>Estimated average yield</small><h2>${Number(d.prediction).toLocaleString(undefined,{maximumFractionDigits:3})}</h2><p>Ensemble of three regression models using simple farmer inputs.</p></div><div class="harvest-badge"><small>For ${d.area_acres} acres</small><b>${Number(d.estimated_harvest).toLocaleString(undefined,{maximumFractionDigits:3})}</b><small>estimated harvest</small></div></div>
    ${modelRegressionPanel(d.models)}
    <div class="dashboard-grid two"><div class="analysis-card"><div class="card-heading"><span>📊</span><div><b>Input analysis</b><small>Simple interpretation of the information you gave.</small></div></div>${bars(d.analysis)}</div>
    <div class="analysis-card"><div class="card-heading"><span>🏡</span><div><b>Local reference</b><small>${esc(d.baseline_source)}</small></div></div><div class="baseline-box">${d.baseline!==null?`<span>Historical median yield</span><b>${Number(d.baseline).toLocaleString(undefined,{maximumFractionDigits:3})}</b><small>Compare this reference with the AI estimate above.</small>`:`<span>No local baseline available</span><small>The AI result is based on the wider training data.</small>`}</div><p class="muted">${esc(d.explanation)}</p></div></div>
    <div class="analysis-card"><div class="card-heading"><span>📉</span><div><b>Model comparison</b><small>Closer model values mean stronger agreement.</small></div></div>${regressionBars(d.models,false)}</div>`;
  }catch(err){box.innerHTML=`<div class="error-card">⚠️ ${esc(err.message)}</div>`;}
});

const priceForm=document.getElementById('priceForm');
if(priceForm) priceForm.addEventListener('submit',async e=>{
  e.preventDefault(); const box=document.getElementById('priceResult'); setLoading(box,'Reading the historical market pattern…');
  try{
    const d=await post('/api/price-forecast',priceForm);
    const direction=d.prediction>=d.last_price?'up':'down';
    box.innerHTML=`<div class="result-hero"><div class="result-crop-icon">${cropEmoji(d.commodity)}</div><div><small>Forecast for ${esc(d.forecast_month)}</small><h2>₹${money(d.prediction)}</h2><p>Latest supplied dataset month: ${esc(d.last_month)} • current dataset-based reference ₹${money(d.last_price)}</p></div><div class="trend-badge ${direction}"><b>${d.prediction>=d.last_price?'↗':'↘'} ${((d.prediction-d.last_price)/d.last_price*100).toFixed(1)}%</b><small>vs latest dataset price</small></div></div>
    ${modelRegressionPanel(d.models.map((x,i)=>({...x,name:x.name,value:x.value})))}
    <div class="dashboard-grid two"><div class="analysis-card"><div class="card-heading"><span>📈</span><div><b>12-month price movement</b><small>Historical values from the supplied dataset.</small></div></div>${lineChart(d.history)}</div>
    <div class="analysis-card"><div class="card-heading"><span>📊</span><div><b>Forecast analysis</b><small>Use the bars to compare stability and movement.</small></div></div>${bars(d.analysis)}<div class="model-note">ℹ️ ${esc(d.note)}</div></div></div>
    <div class="analysis-card"><div class="card-heading"><span>🧺</span><div><b>How to read this</b><small>Two models estimate the next month, then the app averages them.</small></div></div>${regressionBars(d.models,true)}</div>`;
  }catch(err){box.innerHTML=`<div class="error-card">⚠️ ${esc(err.message)}</div>`;}
});

const GiB = 1024 ** 3;

const MODELS = [
  { id:'qwen3-32b', name:'Qwen3 32B', family:'Qwen', params:32.8, active:32.8, layers:64, heads:64, kvHeads:8, headDim:128, maxContext:131072, attentionFactor:1, cacheType:'GQA', confidence:'verified', source:'curated architecture profile' },
  { id:'qwen3-30b-a3b', name:'Qwen3 30B-A3B', family:'Qwen', params:30.5, active:3.3, layers:48, heads:32, kvHeads:4, headDim:128, maxContext:131072, attentionFactor:1, cacheType:'MoE + GQA', confidence:'verified', source:'curated architecture profile' },
  { id:'qwen3-14b', name:'Qwen3 14B', family:'Qwen', params:14.8, active:14.8, layers:40, heads:40, kvHeads:8, headDim:128, maxContext:131072, attentionFactor:1, cacheType:'GQA', confidence:'verified', source:'curated architecture profile' },
  { id:'qwen25-72b', name:'Qwen2.5 72B Instruct', family:'Qwen', params:72.7, active:72.7, layers:80, heads:64, kvHeads:8, headDim:128, maxContext:131072, attentionFactor:1, cacheType:'GQA', confidence:'verified', source:'curated architecture profile' },
  { id:'glm45', name:'GLM-4.5 355B-A32B', family:'GLM', params:355, active:32, layers:92, heads:96, kvHeads:8, headDim:128, maxContext:131072, attentionFactor:1, cacheType:'MoE + estimated GQA', confidence:'estimated', source:'curated estimate', note:'Confirm KV geometry with the exact repository config.' },
  { id:'glm45-air', name:'GLM-4.5 Air 106B-A12B', family:'GLM', params:106, active:12, layers:46, heads:64, kvHeads:8, headDim:128, maxContext:131072, attentionFactor:1, cacheType:'MoE + estimated GQA', confidence:'estimated', source:'curated estimate', note:'Confirm KV geometry with the exact repository config.' },
  { id:'deepseek-r1', name:'DeepSeek R1 / V3 671B-A37B', family:'DeepSeek', params:671, active:37, layers:61, heads:128, kvHeads:1, headDim:576, maxContext:131072, customKvPerToken:70272, cacheType:'MoE + MLA', confidence:'estimated', source:'MLA cache estimate', note:'MLA cache representation varies by engine; verify after engine initialization.' },
  { id:'deepseek-r1-distill-32b', name:'DeepSeek R1 Distill Qwen 32B', family:'DeepSeek', params:32.8, active:32.8, layers:64, heads:40, kvHeads:8, headDim:128, maxContext:131072, attentionFactor:1, cacheType:'Dense + GQA', confidence:'verified', source:'curated architecture profile' },
  { id:'gemma3-27b', name:'Gemma 3 27B', family:'Gemma', params:27.4, active:27.4, layers:62, heads:32, kvHeads:16, headDim:128, maxContext:131072, attentionFactor:.25, slidingWindow:1024, cacheType:'Hybrid local/global attention', confidence:'estimated', source:'hybrid-cache approximation', note:'Local sliding-window layers are approximated; runtime cache implementations differ.' },
  { id:'gemma3-12b', name:'Gemma 3 12B', family:'Gemma', params:12, active:12, layers:48, heads:16, kvHeads:8, headDim:256, maxContext:131072, attentionFactor:.25, slidingWindow:1024, cacheType:'Hybrid local/global attention', confidence:'estimated', source:'hybrid-cache approximation' },
  { id:'llama4-scout', name:'Llama 4 Scout 109B-A17B', family:'Llama', params:109, active:17, layers:48, heads:40, kvHeads:8, headDim:128, maxContext:1000000, attentionFactor:1, cacheType:'MoE + GQA', confidence:'estimated', source:'curated estimate', note:'Million-token operation needs engine-specific cache and RoPE validation.' },
  { id:'mistral-small-31', name:'Mistral Small 3.1 24B', family:'Mistral', params:24, active:24, layers:40, heads:32, kvHeads:8, headDim:128, maxContext:131072, attentionFactor:1, cacheType:'Dense + GQA', confidence:'verified', source:'curated architecture profile' },
  { id:'gpt-oss-120b', name:'gpt-oss 120B-A5.1B', family:'OpenAI', params:117, active:5.1, layers:36, heads:64, kvHeads:8, headDim:128, maxContext:131072, attentionFactor:1, cacheType:'MoE + GQA estimate', confidence:'estimated', source:'curated estimate' },
  { id:'gpt-oss-20b', name:'gpt-oss 20B-A3.6B', family:'OpenAI', params:21, active:3.6, layers:24, heads:32, kvHeads:8, headDim:128, maxContext:131072, attentionFactor:1, cacheType:'MoE + GQA estimate', confidence:'estimated', source:'curated estimate' },
  { id:'custom', name:'Custom model · registry lookup', family:'Custom', params:27, active:27, layers:48, heads:32, kvHeads:8, headDim:128, maxContext:32768, attentionFactor:1, cacheType:'Assumed GQA', confidence:'estimated', source:'editable default', note:'Fetch a repository config for a better estimate.' }
];

const GPUS = [
  {id:'auto', name:'Auto-select best fit', vendor:'Any', vram:0, bandwidth:0, bf16:0, fp8:0, power:0, price:0, link:'auto'},
  {id:'rtx3090', name:'RTX 3090', vendor:'NVIDIA', vram:24, bandwidth:936, bf16:71, fp8:0, power:350, price:1.0, link:'pcie'},
  {id:'rtx4090', name:'RTX 4090', vendor:'NVIDIA', vram:24, bandwidth:1008, bf16:82.6, fp8:165, power:450, price:1.35, link:'pcie'},
  {id:'rtx5090', name:'RTX 5090', vendor:'NVIDIA', vram:32, bandwidth:1792, bf16:105, fp8:210, power:575, price:1.8, link:'pcie'},
  {id:'a6000', name:'RTX A6000', vendor:'NVIDIA', vram:48, bandwidth:768, bf16:77.4, fp8:0, power:300, price:2.2, link:'nvlink'},
  {id:'l40s', name:'L40S', vendor:'NVIDIA', vram:48, bandwidth:864, bf16:362, fp8:733, power:350, price:2.8, link:'pcie'},
  {id:'a100-80', name:'A100 80GB', vendor:'NVIDIA', vram:80, bandwidth:2039, bf16:312, fp8:0, power:400, price:4.2, link:'nvlink'},
  {id:'a800-80', name:'A800 80GB', vendor:'NVIDIA', vram:80, bandwidth:2039, bf16:312, fp8:0, power:400, price:3.8, link:'nvlink'},
  {id:'h20', name:'H20 96GB', vendor:'NVIDIA', vram:96, bandwidth:4000, bf16:148, fp8:296, power:400, price:4.4, link:'nvlink'},
  {id:'h100-80', name:'H100 80GB', vendor:'NVIDIA', vram:80, bandwidth:3350, bf16:989, fp8:1979, power:700, price:6.0, link:'nvlink'},
  {id:'h200', name:'H200 141GB', vendor:'NVIDIA', vram:141, bandwidth:4800, bf16:989, fp8:1979, power:700, price:7.2, link:'nvlink'},
  {id:'b200', name:'B200 192GB', vendor:'NVIDIA', vram:192, bandwidth:8000, bf16:2250, fp8:4500, power:1000, price:10, link:'nvlink'},
  {id:'mi300x', name:'AMD MI300X 192GB', vendor:'AMD', vram:192, bandwidth:5300, bf16:1307, fp8:2614, power:750, price:6.5, link:'rocm'}
];

const QUANTS = {
  bf16:{name:'BF16 / FP16', bytes:2, metadata:1.025, compute:'bf16', quality:'native'},
  fp8:{name:'FP8 weights', bytes:1, metadata:1.07, compute:'fp8', quality:'near-native'},
  int8:{name:'INT8 / W8A8', bytes:1, metadata:1.09, compute:'fp8', quality:'high'},
  int4:{name:'INT4 / AWQ / GPTQ', bytes:.5, metadata:1.18, compute:'int4', quality:'compressed'},
  q6:{name:'GGUF Q6_K', bytes:.75, metadata:1.12, compute:'bf16', quality:'high'},
  q4:{name:'GGUF Q4_K_M', bytes:.5, metadata:1.22, compute:'int4', quality:'compressed'}
};
const KV_TYPES = {bf16:{name:'BF16 / FP16',bytes:2},fp8:{name:'FP8',bytes:1},int8:{name:'INT8',bytes:1}};

let currentModel = MODELS[0];
let lastResult = null;
let savedScenarios = [];
try { savedScenarios = JSON.parse(localStorage.getItem('tensorplan-scenarios') || '[]'); } catch (_) { savedScenarios = []; }
const $ = id => document.getElementById(id);
const fmt = (n,d=1) => Number(n).toLocaleString(undefined,{maximumFractionDigits:d});
const gib = bytes => bytes / GiB;
const escapeHtml = value => String(value).replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));

function populate() {
  const grouped = MODELS.reduce((a,m)=>{ if(!a[m.family]) a[m.family]=[]; a[m.family].push(m); return a; },{});
  $('modelSelect').innerHTML = Object.entries(grouped).map(([family,models])=>`<optgroup label="${family}">${models.map(m=>`<option value="${m.id}">${m.name}</option>`).join('')}</optgroup>`).join('');
  $('gpuSelect').innerHTML = GPUS.map(g=>`<option value="${g.id}">${g.name}${g.vram?` · ${g.vram}GB`:''}</option>`).join('');
  $('quantSelect').innerHTML = Object.entries(QUANTS).map(([id,q])=>`<option value="${id}">${q.name}</option>`).join('');
  $('kvSelect').innerHTML = Object.entries(KV_TYPES).map(([id,q])=>`<option value="${id}">${q.name}</option>`).join('');
  $('contextSelect').innerHTML = [2048,4096,8192,16384,32768,65536,131072,262144,1000000].map(n=>`<option value="${n}" ${n===32768?'selected':''}>${n>=1000000?'1M':n>=1024?`${n/1024}K`:n} tokens</option>`).join('');
  $('quantSelect').value='int4';
  updateModelSummary();
}

function updateModelSummary() {
  const m=currentModel;
  $('modelSummary').innerHTML=`<div><b>${fmt(m.params)}B</b><span>Total params</span></div><div><b>${fmt(m.active)}B</b><span>Active / token</span></div><div><b>${m.cacheType}</b><span>Cache design</span></div>`;
  $('confidenceBadge').textContent = m.confidence==='verified'?'HIGH-CONFIDENCE PROFILE':'ESTIMATED PROFILE';
  $('confidenceBadge').classList.toggle('exact',m.confidence==='verified');
}

function kvBytesPerToken(m,kvBytes) {
  if (m.customKvPerToken) return m.customKvPerToken * (kvBytes/2);
  return 2 * m.layers * m.kvHeads * m.headDim * kvBytes * (m.attentionFactor == null ? 1 : m.attentionFactor);
}

function topologyEfficiency(n,link) {
  if(n===1) return 1;
  const base = link==='nvlink'||link==='rocm' ? .94 : .82;
  const nodePenalty = n>8 ? .88 : 1;
  return Math.pow(base,Math.log2(n)) * nodePenalty;
}

function runtimeEfficiency(runtime,traffic) {
  const r={vllm:.61,sglang:.64,trtllm:.70,llamacpp:.43,generic:.50}[runtime];
  const t={interactive:.82,balanced:.92,batch:1}[traffic];
  return r*t;
}

function performance(m,gpu,n,weightBytes,kvToken,inputs) {
  const link=inputs.link==='auto'?gpu.link:inputs.link;
  const topo=topologyEfficiency(n,link);
  const eff=runtimeEfficiency(inputs.runtime,inputs.traffic);
  const batchBase={interactive:.62,balanced:.78,batch:.92}[inputs.traffic];
  const effectiveBatch=Math.max(1,Math.min(inputs.concurrency,inputs.traffic==='batch'?32:16)*batchBase);
  let tensor=gpu.bf16;
  if(QUANTS[inputs.quant].compute==='fp8') tensor=gpu.fp8||gpu.bf16*1.35;
  if(QUANTS[inputs.quant].compute==='int4') tensor=(gpu.fp8||gpu.bf16)*1.65;
  const opsPerToken=2*m.active*1e9;
  const compute=tensor*1e12*n*eff*topo/opsPerToken;
  const cacheRead=kvToken*Math.max(1,inputs.avgContext);
  const bytesPerGeneratedToken=weightBytes/effectiveBatch+cacheRead;
  const bandwidth=gpu.bandwidth*1e9*n*eff*topo/bytesPerGeneratedToken;
  const decode=Math.max(0,Math.min(compute,bandwidth));
  return {decode,compute,bandwidth,effectiveBatch,topo,link,bottleneck:compute<bandwidth?'compute':'memory bandwidth'};
}

function sizeForGpu(m,gpu,inputs) {
  const q=QUANTS[inputs.quant], kv=KV_TYPES[inputs.kv];
  const rawWeight=m.params*1e9*q.bytes;
  const weights=rawWeight*q.metadata;
  const kvToken=kvBytesPerToken(m,kv.bytes);
  const recurrentState=(m.stateBytesPerSequence||0)*inputs.concurrency;
  const cache=kvToken*inputs.context*inputs.concurrency+recurrentState;
  const evaluate=n=>{
    const runtime=Math.max(4*GiB*n,weights*.055+cache*.035)+Math.max(0,n-1)*.6*GiB;
    const total=weights+cache+runtime;
    const capacity=gpu.vram*GiB*n*inputs.util;
    const perf=performance(m,gpu,n,weights,kvToken,inputs);
    const meetsMemory=total<=capacity;
    const meetsTps=!inputs.targetTps||perf.decode>=inputs.targetTps;
    return {runtime,total,capacity,perf,meetsMemory,meetsTps};
  };
  const required=TensorPlanConfig.findRequiredGpuCount(inputs.maxGpu,evaluate);
  return {...required,weights,rawWeight,cache,kvToken,gpu,recurrentState,unmet:required.exceedsMax};
}

function getInputs(){return {
  quant:$('quantSelect').value,kv:$('kvSelect').value,context:+$('contextSelect').value,concurrency:Math.max(1,+$('concurrencyInput').value||1),
  avgContext:Math.max(1,+$('avgContextInput').value||1),output:Math.max(1,+$('outputInput').value||1),runtime:$('runtimeSelect').value,
  util:TensorPlanConfig.normalizeUtilization($('utilSelect').value,.90),targetTps:Math.max(0,+$('targetTpsInput').value||0),traffic:$('trafficSelect').value,link:$('linkSelect').value,maxGpu:+$('maxGpuSelect').value
}}

function rankGpus(m,inputs){
  return GPUS.filter(g=>g.id!=='auto').map(g=>{
    const r=sizeForGpu(m,g,inputs);
    const waste=Math.max(0,r.capacity-r.total)/Math.max(r.capacity,1);
    const perfFit=inputs.targetTps?Math.min(1,r.perf.decode/inputs.targetTps):Math.min(1,r.perf.decode/Math.max(20,inputs.concurrency*12));
    const multiGpuPenalty=5*Math.max(0,r.n-1)+(g.link==='pcie'&&r.n>1?15:0);
    const score=(r.unmet?-100:0)+42*(1-waste)+33*perfFit+20/(g.price*r.n)+5*(g.link!=='pcie'||r.n===1)-multiGpuPenalty;
    return {...r,score};
  }).sort((a,b)=>b.score-a.score);
}

function calculate(){
  const inputs=getInputs();
  if(inputs.avgContext>inputs.context){inputs.avgContext=inputs.context;$('avgContextInput').value=inputs.context}
  const ranked=rankGpus(currentModel,inputs);
  const selectedId=$('gpuSelect').value;
  const result=selectedId==='auto'?ranked.find(r=>!r.unmet)||ranked[0]:sizeForGpu(currentModel,GPUS.find(g=>g.id===selectedId),inputs);
  render(result,ranked,inputs); lastResult={model:currentModel,inputs,result};
}

function render(r,ranked,inputs){
  const m=currentModel,g=r.gpu,fit=Math.min(100,r.total/r.capacity*100),perUser=r.perf.decode/inputs.concurrency;
  $('resultTitle').textContent=m.name;
  $('neededGpu').textContent=r.unresolved?`>${r.n}`:r.n;
  $('topologyText').textContent=`${r.n} × ${g.name}`;
  $('topologyNote').textContent=`${r.perf.link.toUpperCase()} · ${inputs.runtime.toUpperCase()} · ${QUANTS[inputs.quant].name}`;
  $('fitBar').style.width=`${Math.min(100,fit)}%`;
  $('fitBar').style.background=fit>95?'var(--orange)':'var(--green)';
  $('fitText').textContent=r.unresolved?`Requirement exceeds calculation range`:r.unmet?`${r.n} GPUs required; configured maximum is ${inputs.maxGpu}`:`${fmt(fit)}% of usable cluster VRAM · ${fmt(gib(r.capacity-r.total))} GiB reserve`;
  $('configSummary').textContent=`${QUANTS[inputs.quant].name} · ${fmt(inputs.context,0)} × ${inputs.concurrency}`;
  $('weightsMetric').textContent=`${fmt(gib(r.weights))} GiB`;
  $('weightsDetail').textContent=`${fmt(gib(r.rawWeight))} GiB raw + metadata`;
  $('kvMetric').textContent=`${fmt(gib(r.cache))} GiB`;
  $('kvDetail').textContent=(r.recurrentState||0)>0?`${fmt(r.kvToken/1024)} KiB/token + ${fmt((m.stateBytesPerSequence||0)/1048576)} MiB/request`:`${fmt(r.kvToken/1024)} KiB/token × ${inputs.concurrency}`;
  $('tpsMetric').textContent=`~${fmt(r.perf.decode,0)} tok/s`;
  $('tpsDetail').textContent=`aggregate · ${r.perf.bottleneck}-bound`;
  $('userTpsMetric').textContent=`~${fmt(perUser,1)} tok/s`;
  $('userTpsDetail').textContent=`fair-share at ${inputs.concurrency} concurrent`;
  $('memoryTotal').textContent=`${fmt(gib(r.total))} GiB / ${fmt(gib(r.capacity))} GiB usable`;
  const items=[['Weights',r.weights,'var(--ink)'],['KV cache',r.cache,'var(--green-dark)'],['Runtime',r.runtime,'var(--orange)'],['Reserve',Math.max(0,r.capacity-r.total),'#cbd0c7']];
  $('memoryBar').innerHTML=items.map(x=>`<i style="width:${Math.max(0,x[1]/r.capacity*100)}%;background:${x[2]}"></i>`).join('');
  $('memoryLegend').innerHTML=items.map(x=>`<div style="--c:${x[2]}"><span>${x[0]}</span><b>${fmt(gib(x[1]))} GiB</b></div>`).join('');
  $('memoryFormula').textContent=`${fmt(m.params)}B × ${QUANTS[inputs.quant].bytes} byte × metadata + ${fmt(r.kvToken/1024)} KiB/token × ${fmt(inputs.context,0)} × ${inputs.concurrency}${(r.recurrentState||0)>0?` + ${fmt((m.stateBytesPerSequence||0)/1048576)} MiB state × ${inputs.concurrency}`:''} + runtime`;
  $('computeLimit').textContent=`${fmt(r.perf.compute,0)} tok/s`;
  $('bandwidthLimit').textContent=`${fmt(r.perf.bandwidth,0)} tok/s`;
  const maxLimit=Math.max(r.perf.compute,r.perf.bandwidth,1);
  $('computeBar').style.width=`${r.perf.compute/maxLimit*100}%`; $('bandwidthBar').style.width=`${r.perf.bandwidth/maxLimit*100}%`;
  $('bottleneckText').innerHTML=`Estimated decode is <strong>${r.perf.bottleneck}-bound</strong>. Effective continuous batch: ${fmt(r.perf.effectiveBatch,1)}. Parallel scaling factor: ${fmt(r.perf.topo*100,0)}%.`;
  $('bottleneckBadge').textContent=`${r.perf.bottleneck}-bound`;
  $('recommendationList').innerHTML=ranked.slice(0,4).map((x,i)=>`<div class="recommendation"><span class="rank">0${i+1}</span><div class="rec-name"><strong>${x.n} × ${x.gpu.name}</strong><span>${x.gpu.vendor} · ${x.gpu.vram}GB each</span></div><div class="rec-stat"><strong>${fmt(gib(x.capacity-x.total))} GiB</strong><span>Reserve</span></div><div class="rec-stat"><strong>~${fmt(x.perf.decode,0)} tok/s</strong><span>Decode</span></div><div class="score">${Math.max(0,fmt(x.score,0))}<i style="width:${Math.max(4,Math.min(100,x.score))}%"></i></div></div>`).join('');
  const details=[
    ['Total / active parameters',`${fmt(m.params)}B / ${fmt(m.active)}B`],['Layers / KV heads / head dim',`${m.layers} / ${m.kvHeads} / ${m.headDim}`],
    ['Cache architecture',m.cacheType],['KV bytes per token',`${fmt(r.kvToken/1024)} KiB`],['Recurrent state cache',`${fmt(gib(r.recurrentState||0),2)} GiB`],['Maximum reserved tokens',fmt(inputs.context*inputs.concurrency,0)],
    ['Weight precision',QUANTS[inputs.quant].name],['KV-cache precision',KV_TYPES[inputs.kv].name],['Usable VRAM policy',`${inputs.util*100}%`],
    ['Runtime overhead',`${fmt(gib(r.runtime))} GiB`],['Interconnect efficiency',`${fmt(r.perf.topo*100,0)}%`],['Compute ceiling',`${fmt(r.perf.compute,0)} tok/s`],
    ['Bandwidth ceiling',`${fmt(r.perf.bandwidth,0)} tok/s`],['Est. generation time',`${fmt(inputs.output/Math.max(perUser,.01),1)} sec / request`],
    ['Cluster board power',`${fmt(g.power*r.n,0)} W TDP`],['Profile source',m.source],['Configuration confidence',m.confidence]
  ];
  $('detailTable').innerHTML=details.map(x=>`<div><span>${x[0]}</span><code>${x[1]}</code></div>`).join('');
  const warnings=[];
  if(r.unresolved) warnings.push(`The requirement is greater than the calculator's ${r.n}-GPU safety range.`);
  else if(r.unmet) warnings.push(`This workload requires ${r.n} × ${g.name}, exceeding the configured maximum of ${inputs.maxGpu}. The maximum is treated as a planning constraint, not as the reported answer.`);
  if(inputs.context>m.maxContext) warnings.push(`Configured context exceeds this profile's advertised ${fmt(m.maxContext,0)}-token context. Engine and RoPE configuration must be validated.`);
  if(r.n>1&&r.perf.link==='pcie') warnings.push('Tensor parallelism over PCIe may deliver lower throughput and higher tail latency than this estimate, especially on consumer GPUs.');
  if(m.note) warnings.push(m.note);
  if(inputs.quant==='int4'||inputs.quant==='q4') warnings.push('4-bit quality and speed depend on the quantization method and whether the selected GPU has an optimized kernel.');
  warnings.push('Throughput is a planning estimate. Benchmark the exact checkpoint, runtime, prompt distribution and sampling settings before procurement.');
  $('warnings').innerHTML=warnings.map((w,i)=>`<div class="warning ${i===warnings.length-1?'info':''}">${w}</div>`).join('');
}

async function fetchJson(url,timeout=9000){
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),timeout);
  try{const res=await fetch(url,{signal:controller.signal,headers:{Accept:'application/json'}});if(!res.ok)throw new Error(`${res.status}`);return await res.json()}finally{clearTimeout(timer)}
}

async function fetchText(url,timeout=5000){
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),timeout);
  try{const res=await fetch(url,{signal:controller.signal,headers:{Accept:'text/plain,text/markdown'}});if(!res.ok)throw new Error(`${res.status}`);return await res.text()}finally{clearTimeout(timer)}
}

async function lookup(){
  const id=$('modelIdInput').value.trim().replace(/^https?:\/\/(huggingface\.co|modelscope\.cn\/models)\//,'').replace(/\/$/,'');
  if(!id||!id.includes('/')){setLookup('Enter an owner/model repository ID.',true);return}
  $('lookupBtn').disabled=true; setLookup('Checking ModelScope…');
  const sources=[
    ['ModelScope',`https://modelscope.cn/models/${id}/resolve/master/config.json`,`https://modelscope.cn/models/${id}/resolve/master/README.md`],
    ['ModelScope',`https://www.modelscope.cn/models/${id}/resolve/master/config.json`,`https://www.modelscope.cn/models/${id}/resolve/master/README.md`],
    ['Hugging Face',`https://huggingface.co/${id}/resolve/main/config.json`,`https://huggingface.co/${id}/resolve/main/README.md`]
  ];
  let error;
  for(const [source,url,readmeUrl] of sources){
    try{
      if(source==='Hugging Face')setLookup('ModelScope unavailable. Trying Hugging Face…');
      const config=await fetchJson(url);
      try{
        const counts=TensorPlanConfig.modelCardCounts(await fetchText(readmeUrl));
        if(counts.total_parameters||counts.activated_parameters)config.model_card_counts=counts;
      }catch(e){}
      currentModel=TensorPlanConfig.normalizeConfig(id,config,source); MODELS.splice(MODELS.length-1,0,currentModel);
      const option=document.createElement('option');option.value=currentModel.id;option.textContent=`${currentModel.name} · live ${source}`;$('modelSelect').appendChild(option);$('modelSelect').value=currentModel.id;
      updateModelSummary();calculate();setLookup(`Loaded ${source} config · ${currentModel.layers} layers · ${fmt(currentModel.params)}B parameters · ${currentModel.parameterMethod}`);toast('Registry profile loaded');$('lookupBtn').disabled=false;return;
    }catch(e){error=e}
  }
  setLookup(`Lookup failed (${error && error.message || 'network/CORS'}). Try another public repository ID.`,true);$('lookupBtn').disabled=false;
}

function setLookup(text,error=false){$('lookupStatus').textContent=text;$('lookupStatus').style.color=error?'#b42318':'#65717c'}
function toast(message){const el=$('toast');el.textContent=message;el.classList.add('show');setTimeout(()=>el.classList.remove('show'),2200)}
function copyDetails(){if(!lastResult)return;const r=lastResult.result,i=lastResult.inputs,m=lastResult.model;const text=`TensorPlan deployment estimate\nModel: ${m.name}\nHardware: ${r.n} x ${r.gpu.name}\nQuantization: ${QUANTS[i.quant].name}\nContext x concurrency: ${fmt(i.context,0)} x ${i.concurrency}\nWeights: ${fmt(gib(r.weights))} GiB\nKV cache: ${fmt(gib(r.cache))} GiB\nRuntime: ${fmt(gib(r.runtime))} GiB\nUsable capacity: ${fmt(gib(r.capacity))} GiB\nEstimated decode: ${fmt(r.perf.decode,0)} tok/s\nBottleneck: ${r.perf.bottleneck}\nProfile confidence: ${m.confidence}`;navigator.clipboard.writeText(text).then(()=>toast('Calculation copied'))}
function exportPlan(){if(!lastResult)return;const clean={generatedAt:new Date().toISOString(),disclaimer:'Planning estimate; benchmark before procurement.',model:{...lastResult.model,rawConfig:undefined},inputs:lastResult.inputs,result:{gpu:lastResult.result.gpu.name,gpuCount:lastResult.result.n,weightsGiB:gib(lastResult.result.weights),kvCacheGiB:gib(lastResult.result.cache),runtimeGiB:gib(lastResult.result.runtime),usableCapacityGiB:gib(lastResult.result.capacity),estimatedDecodeTps:lastResult.result.perf.decode,bottleneck:lastResult.result.perf.bottleneck}};const blob=new Blob([JSON.stringify(clean,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`tensorplan-${currentModel.id}.json`;a.click();URL.revokeObjectURL(a.href);toast('Plan exported')}
function reset(){
  $('modelSelect').value='qwen3-32b';currentModel=MODELS[0];$('quantSelect').value='int4';$('kvSelect').value='bf16';$('contextSelect').value='32768';$('concurrencyInput').value=4;$('avgContextInput').value=8192;$('outputInput').value=512;$('runtimeSelect').value='vllm';$('utilSelect').value='0.90';$('targetTpsInput').value=0;$('trafficSelect').value='interactive';$('gpuSelect').value='auto';$('linkSelect').value='auto';$('maxGpuSelect').value=8;$('modelIdInput').value='';setLookup('ModelScope, then Hugging Face fallback');document.querySelectorAll('.preset').forEach(b=>b.classList.remove('active'));updateModelSummary();calculate();toast('Planner reset')
}

function applyPreset(name){
  const presets={
    chat:{context:8192,concurrency:4,avg:4096,output:512,traffic:'interactive',target:0},
    long:{context:131072,concurrency:1,avg:65536,output:1024,traffic:'interactive',target:0},
    api:{context:16384,concurrency:16,avg:8192,output:512,traffic:'balanced',target:200},
    batch:{context:8192,concurrency:64,avg:4096,output:256,traffic:'batch',target:0}
  };
  const p=presets[name]; if(!p)return;
  $('contextSelect').value=String(p.context);$('concurrencyInput').value=p.concurrency;$('avgContextInput').value=p.avg;$('outputInput').value=p.output;$('trafficSelect').value=p.traffic;$('targetTpsInput').value=p.target;
  document.querySelectorAll('.preset').forEach(b=>b.classList.toggle('active',b.dataset.preset===name));
  calculate();
}

function openModelEditor(){
  $('editParams').value=currentModel.params;$('editActive').value=currentModel.active;$('editLayers').value=currentModel.layers;$('editHeads').value=currentModel.heads;$('editKvHeads').value=currentModel.kvHeads;$('editHeadDim').value=currentModel.headDim;$('editAttentionFactor').value=currentModel.attentionFactor==null?1:currentModel.attentionFactor;$('editMaxContext').value=currentModel.maxContext;
  $('modelDialog').showModal();
}

function applyModelOverride(event){
  event.preventDefault();
  const overridden={...currentModel,id:`override-${Date.now()}`,name:`${currentModel.name} · override`,params:+$('editParams').value,active:+$('editActive').value,layers:+$('editLayers').value,heads:+$('editHeads').value,kvHeads:+$('editKvHeads').value,headDim:+$('editHeadDim').value,attentionFactor:+$('editAttentionFactor').value,maxContext:+$('editMaxContext').value,customKvPerToken:null,confidence:'estimated',source:'session override',note:'Model geometry was manually overridden for this estimate.'};
  currentModel=overridden;MODELS.splice(MODELS.length-1,0,overridden);
  const option=document.createElement('option');option.value=overridden.id;option.textContent=overridden.name;$('modelSelect').appendChild(option);$('modelSelect').value=overridden.id;
  $('modelDialog').close();updateModelSummary();calculate();toast('Model override applied');
}

function saveScenario(){
  if(!lastResult)return;const r=lastResult.result,i=lastResult.inputs,m=lastResult.model;
  savedScenarios.push({id:Date.now(),model:m.name,quant:QUANTS[i.quant].name,workload:`${fmt(i.context,0)} × ${i.concurrency}`,hardware:`${r.n} × ${r.gpu.name}`,memory:`${fmt(gib(r.total))} GiB`,tps:`${fmt(r.perf.decode,0)} tok/s`});
  if(savedScenarios.length>8)savedScenarios.shift();localStorage.setItem('tensorplan-scenarios',JSON.stringify(savedScenarios));renderScenarios();toast('Scenario saved');
}

function renderScenarios(){
  $('scenarioEmpty').style.display=savedScenarios.length?'none':'block';
  if(!savedScenarios.length){$('scenarioTable').innerHTML='';return}
  $('scenarioTable').innerHTML=`<div class="scenario-row header"><span>Model / precision</span><span>Workload</span><span>Hardware</span><span>Memory</span><span>Decode</span><span></span></div>`+savedScenarios.map(s=>`<div class="scenario-row"><span><strong>${escapeHtml(s.model)}</strong><br><code>${escapeHtml(s.quant)}</code></span><code>${escapeHtml(s.workload)}</code><span>${escapeHtml(s.hardware)}</span><code>${escapeHtml(s.memory)}</code><code>${escapeHtml(s.tps)}</code><button class="remove-scenario" data-remove-scenario="${escapeHtml(s.id)}" title="Remove scenario">×</button></div>`).join('');
}

function clearScenarios(){savedScenarios=[];localStorage.removeItem('tensorplan-scenarios');renderScenarios();toast('Saved scenarios cleared')}
function bind(){
  document.querySelectorAll('select,input[type=number]').forEach(el=>el.addEventListener('input',()=>{if(el.id==='modelSelect'){currentModel=MODELS.find(m=>m.id===el.value);updateModelSummary();const ctx=Math.min(32768,currentModel.maxContext);$('contextSelect').value=String(ctx)}calculate()}));
  $('lookupBtn').addEventListener('click',lookup);$('modelIdInput').addEventListener('keydown',e=>{if(e.key==='Enter')lookup()});$('copyDetailsBtn').addEventListener('click',copyDetails);$('exportBtn').addEventListener('click',exportPlan);$('resetBtn').addEventListener('click',reset);
  document.querySelectorAll('.preset').forEach(button=>button.addEventListener('click',()=>applyPreset(button.dataset.preset)));
  $('editModelBtn').addEventListener('click',openModelEditor);$('applyModelConfigBtn').addEventListener('click',applyModelOverride);$('saveScenarioBtn').addEventListener('click',saveScenario);$('clearScenariosBtn').addEventListener('click',clearScenarios);
  $('scenarioTable').addEventListener('click',event=>{const id=event.target.dataset.removeScenario;if(!id)return;savedScenarios=savedScenarios.filter(s=>String(s.id)!==id);localStorage.setItem('tensorplan-scenarios',JSON.stringify(savedScenarios));renderScenarios()});
}

populate();bind();renderScenarios();calculate();

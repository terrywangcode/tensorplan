(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  if(root)root.TensorPlanConfig=api;
})(typeof window!=='undefined'?window:(typeof global!=='undefined'?global:this),function(){
  function declaredCounts(repositoryId){
    const name=String(repositoryId||'').split('/').pop();
    const totals=[];const totalPattern=/(?:^|[-_])(\d+(?:\.\d+)?)B(?=$|[-_])/gi;let match;
    while((match=totalPattern.exec(name))!==null)totals.push(+match[1]);
    const active=name.match(/(?:^|[-_])A(\d+(?:\.\d+)?)B(?=$|[-_])/i);
    return {total:totals.length?totals[0]:null,active:active?+active[1]:null};
  }

  function languageConfig(config){
    return config.text_config||config.llm_config||config.language_config||config;
  }

  function firstNumber(object,keys,fallback){
    for(const key of keys){
      const value=object&&object[key];
      if(value!==undefined&&value!==null&&value!==''&&Number.isFinite(Number(value)))return Number(value);
    }
    return fallback;
  }

  function parameterBillions(value){
    if(value===undefined||value===null||value==='')return null;
    if(typeof value==='number'&&Number.isFinite(value))return value>1e6?value/1e9:value;
    const match=String(value).trim().replace(/,/g,'').match(/^(\d+(?:\.\d+)?)\s*([KMBT])?(?:\s*(?:parameters?|params?))?$/i);
    if(!match)return null;
    const scale={K:1e3,M:1e6,B:1e9,T:1e12}[String(match[2]||'B').toUpperCase()];
    return Number(match[1])*scale/1e9;
  }

  function publishedCounts(config){
    const c=languageConfig(config);
    const containers=[config,config.model_card_counts,config.model_info,config.parameter_count,c&&c.model_info].filter(Boolean);
    const totalKeys=['num_parameters','total_parameters','total_params','parameter_count','model_size'];
    const activeKeys=['num_active_parameters','active_parameters','activated_parameters','active_params','parameters_per_token'];
    let total=null,active=null;
    for(const object of containers){
      if(total==null)for(const key of totalKeys){total=parameterBillions(object[key]);if(total!=null)break}
      if(active==null)for(const key of activeKeys){active=parameterBillions(object[key]);if(active!=null)break}
    }
    return {total,active};
  }

  function modelCardCounts(markdown){
    const text=String(markdown||'')
      .replace(/<[^>]+>/g,' ')
      .replace(/[`*_#|]/g,' ')
      .replace(/&nbsp;|&#160;/gi,' ')
      .replace(/\s+/g,' ');
    function after(labels){
      const match=text.match(new RegExp(`(?:${labels})\\s*(?:[:=\\-]|is|of)?\\s*(\\d+(?:\\.\\d+)?\\s*[KMBT])\\b`,'i'));
      return match?match[1].replace(/\s+/g,''):null;
    }
    const total=after('total parameters?|parameter count|model parameters?')||((text.match(/\b(\d+(?:\.\d+)?\s*[KMBT])[- ]parameter model\b/i)||[])[1]||null);
    const active=after('activated parameters?|active parameters?|parameters? per token');
    return {total_parameters:total,activated_parameters:active};
  }

  function layerGeometry(c,L){
    const layerTypes=Array.isArray(c.layer_types)?c.layer_types:null;
    const linearConfig=c.linear_attn_config||c.linear_attention_config||null;
    let fullLayers=L,linearLayers=0;
    if(layerTypes){
      fullLayers=layerTypes.filter(x=>['full_attention','global_attention','mla'].includes(String(x).toLowerCase())).length;
      linearLayers=layerTypes.filter(x=>['linear_attention','kda','mamba','ssm'].includes(String(x).toLowerCase())).length;
      fullLayers+=Math.max(0,L-fullLayers-linearLayers);
    }else if(linearConfig){
      const fullList=linearConfig.full_attn_layers||linearConfig.full_attention_layers;
      const linearList=linearConfig.kda_layers||linearConfig.linear_attn_layers||linearConfig.linear_attention_layers;
      if(Array.isArray(fullList)||Array.isArray(linearList)){
        fullLayers=Array.isArray(fullList)?new Set(fullList).size:Math.max(0,L-new Set(linearList).size);
        linearLayers=Array.isArray(linearList)?new Set(linearList).size:Math.max(0,L-fullLayers);
        const unclassified=Math.max(0,L-fullLayers-linearLayers);
        fullLayers+=unclassified;
      }else if(firstNumber(linearConfig,['full_attention_interval'],null)){
        const interval=firstNumber(linearConfig,['full_attention_interval'],1);
        fullLayers=Math.floor(L/interval);linearLayers=L-fullLayers;
      }
    }else if(firstNumber(c,['full_attention_interval'],null)){
      const interval=firstNumber(c,['full_attention_interval'],1);
      fullLayers=Math.floor(L/interval);linearLayers=L-fullLayers;
    }
    return {fullLayers,linearLayers,linearConfig};
  }

  function countMoeLayers(c,L,experts){
    if(experts<=1)return 0;
    const densePrefix=Math.max(0,Math.min(L,firstNumber(c,['first_k_dense_replace','num_dense_layers','first_k_dense_layers'],0)));
    const frequency=Math.max(1,firstNumber(c,['moe_layer_freq','expert_layer_period','moe_frequency'],1));
    let count=0;
    for(let layer=0;layer<L;layer++)if(layer>=densePrefix&&layer%frequency===0)count++;
    return count;
  }

  function conventionalAttention(c,h,heads,kvh,hd){
    const qDim=heads*hd,kvDim=kvh*hd;
    return h*(qDim+2*kvDim)+qDim*h+(c.attn_output_gate?h*qDim:0);
  }

  function mlaAttention(c,h,heads){
    const qRank=firstNumber(c,['q_lora_rank'],null);
    const kvRank=firstNumber(c,['kv_lora_rank'],null);
    const nope=firstNumber(c,['qk_nope_head_dim'],null);
    const rope=firstNumber(c,['qk_rope_head_dim'],0);
    const value=firstNumber(c,['v_head_dim'],null);
    if(kvRank==null||nope==null||value==null)return null;
    const qHead=nope+rope;
    const qParams=qRank==null?h*heads*qHead:h*qRank+qRank*heads*qHead+qRank;
    const kvParams=h*(kvRank+rope)+kvRank+kvRank*heads*(nope+value);
    const output=heads*value*h;
    const gate=c.mla_use_output_gate||c.attn_output_gate?h*heads*value:0;
    return qParams+kvParams+output+gate;
  }

  function linearAttention(c,h,heads,kvh,hd,linearConfig){
    const lc=linearConfig||c;
    const linearHeads=firstNumber(lc,['num_heads'],firstNumber(c,['linear_num_value_heads'],heads));
    const keyHeads=firstNumber(lc,['num_key_heads'],firstNumber(c,['linear_num_key_heads'],kvh));
    const keyDim=linearConfig?firstNumber(lc,['key_head_dim','head_dim'],hd):firstNumber(c,['linear_key_head_dim','key_head_dim'],hd);
    const valueDim=linearConfig?firstNumber(lc,['value_head_dim','head_dim'],hd):firstNumber(c,['linear_value_head_dim','value_head_dim'],hd);
    const qDim=keyHeads*keyDim,kDim=keyHeads*keyDim,vDim=linearHeads*valueDim;
    let params=h*(qDim+kDim+vDim)+vDim*h;
    if(linearConfig){
      const gateRank=firstNumber(lc,['gate_rank','head_dim'],valueDim);
      params+=h*gateRank+gateRank*vDim;
      params+=h*linearHeads;
      params+=(lc.use_full_rank_gate?h*vDim:h*gateRank+gateRank*vDim);
      params+=(qDim+kDim+vDim)*Math.max(0,firstNumber(lc,['short_conv_kernel_size'],1));
      params+=linearHeads+vDim;
    }else params+=h*vDim;
    return params;
  }

  function estimateParams(config){
    const c=languageConfig(config);
    const h=firstNumber(c,['hidden_size','d_model','n_embd'],4096);
    const L=firstNumber(c,['num_hidden_layers','n_layer','num_layers'],32);
    const heads=firstNumber(c,['num_attention_heads','n_head','num_heads'],32);
    const kvh=firstNumber(c,['num_key_value_heads','num_kv_heads','n_kv_head'],heads);
    const hd=firstNumber(c,['head_dim','attention_head_dim','v_head_dim'],firstNumber(c.linear_attn_config||{},['head_dim'],Math.round(h/heads)));
    const inter=firstNumber(c,['intermediate_size','ffn_dim','n_inner'],h*4);
    const vocab=firstNumber(c,['vocab_size','padded_vocab_size'],150000);
    const tie=c.tie_word_embeddings==null?(config.tie_word_embeddings!==false):c.tie_word_embeddings;
    const embeddingParams=vocab*h*(tie?1:2);
    const layers=layerGeometry(c,L);
    const fullAttention=mlaAttention(c,h,heads)||conventionalAttention(c,h,heads,kvh,hd);
    const linearAttentionParams=linearAttention(c,h,heads,kvh,hd,layers.linearConfig);
    const attentionTotal=layers.fullLayers*fullAttention+layers.linearLayers*linearAttentionParams;
    const experts=firstNumber(c,['num_experts','n_routed_experts','num_local_experts'],1);
    const topk=firstNumber(c,['num_experts_per_tok','num_experts_per_token','num_selected_experts','experts_per_token'],experts);
    const expertInter=firstNumber(c,['moe_intermediate_size','expert_intermediate_size','moe_ffn_hidden_size','expert_ffn_dim'],inter);
    const explicitSharedInter=firstNumber(c,['shared_expert_intermediate_size','shared_moe_intermediate_size'],null);
    const shared=firstNumber(c,['num_shared_experts','n_shared_experts','shared_expert_count'],explicitSharedInter==null?0:explicitSharedInter/expertInter);
    const expertInput=firstNumber(c,['routed_expert_hidden_size','latent_moe_dimension','latent_moe_dim','moe_hidden_size'],h);
    const moeLayers=countMoeLayers(c,L,experts),denseLayers=L-moeLayers;
    const mlpPerExpert=3*expertInput*expertInter;
    const sharedMlp=3*h*(explicitSharedInter==null?expertInter*shared:explicitSharedInter);
    const latentProjection=expertInput!==h?2*h*expertInput:0;
    const routerParams=experts>1?h*experts:0;
    const totalMoePerLayer=mlpPerExpert*experts+sharedMlp+latentProjection+routerParams;
    const activeMoePerLayer=mlpPerExpert*Math.min(topk,experts)+sharedMlp+latentProjection+routerParams;
    const denseMlp=3*h*inter;
    const mtpLayers=firstNumber(c,['mtp_num_hidden_layers','num_nextn_predict_layers'],0);
    const mtpParams=mtpLayers*(fullAttention+(experts>1?activeMoePerLayer:denseMlp));
    const residualParams=c.attn_res_block_size?L*(2*h+2*h)+2*h:0;
    const normParams=L*2*h+h;
    const total=embeddingParams+attentionTotal+denseLayers*denseMlp+moeLayers*totalMoePerLayer+mtpParams+residualParams+normParams;
    const active=embeddingParams+attentionTotal+denseLayers*denseMlp+moeLayers*activeMoePerLayer+mtpParams+residualParams+normParams;
    return {total:total/1e9,active:active/1e9,h,L,heads,kvh,hd,fullLayers:layers.fullLayers,linearLayers:layers.linearLayers,moeLayers,denseLayers,experts,topk,shared,expertInput,config:c};
  }

  function cacheGeometry(est){
    const c=est.config;
    let stateBytesPerSequence=0;
    if(est.linearLayers){
      const lc=c.linear_attn_config||c.linear_attention_config||c;
      const keyDim=(c.linear_attn_config||c.linear_attention_config)?firstNumber(lc,['key_head_dim','head_dim'],est.hd):firstNumber(c,['linear_key_head_dim','key_head_dim'],est.hd);
      const valueDim=(c.linear_attn_config||c.linear_attention_config)?firstNumber(lc,['value_head_dim','head_dim'],est.hd):firstNumber(c,['linear_value_head_dim','value_head_dim'],est.hd);
      const valueHeads=firstNumber(lc,['num_heads'],firstNumber(c,['linear_num_value_heads'],est.heads));
      const keyHeads=firstNumber(lc,['num_key_heads'],firstNumber(c,['linear_num_key_heads'],est.kvh));
      const stateBytes=String(c.mamba_ssm_dtype||'float32').includes('32')?4:2;
      stateBytesPerSequence=est.linearLayers*valueHeads*keyDim*valueDim*stateBytes;
      const convChannels=keyHeads*keyDim*2+valueHeads*valueDim;
      const kernel=firstNumber(lc,['short_conv_kernel_size'],firstNumber(c,['linear_conv_kernel_dim'],1));
      stateBytesPerSequence+=est.linearLayers*convChannels*Math.max(0,kernel-1)*2;
    }
    return {attentionFactor:est.fullLayers/est.L,stateBytesPerSequence};
  }

  function normalizeConfig(repositoryId,config,source){
    const est=estimateParams(config);
    const declared=declaredCounts(repositoryId);
    const published=publishedCounts(config);
    const hasExperts=est.experts>1;
    const params=published.total||declared.total||est.total;
    const active=published.active||declared.active||(!hasExperts&&(published.total||declared.total))||est.active;
    const cache=cacheGeometry(est);
    const corroborated=declared.total&&Math.abs(est.total-declared.total)/declared.total<.2;
    const hybrid=est.linearLayers?`Hybrid: ${est.fullLayers} full + ${est.linearLayers} linear`:null;
    const cacheType=[hasExperts?'MoE':null,hybrid||'GQA/MHA'].filter(Boolean).join(' + ');
    const structural=[];
    if(est.expertInput!==est.h)structural.push(`latent experts ${est.expertInput}-wide`);
    if(est.denseLayers&&hasExperts)structural.push(`${est.denseLayers} dense + ${est.moeLayers} MoE layers`);
    if(est.shared)structural.push(`${est.shared} shared experts`);
    if(est.linearLayers)structural.push(`${est.fullLayers} full + ${est.linearLayers} linear attention layers`);
    return {
      id:`remote-${Date.now()}`,name:String(repositoryId).split('/').pop(),family:'Registry',
      params:+params.toFixed(2),active:+active.toFixed(2),layers:est.L,heads:est.heads,kvHeads:est.kvh,headDim:est.hd,
      maxContext:est.config.max_position_embeddings||est.config.seq_length||est.config.model_max_length||32768,
      attentionFactor:cache.attentionFactor,stateBytesPerSequence:cache.stateBytesPerSequence,cacheType,
      confidence:(published.total||corroborated)?'verified':'estimated',source:`${source} config.json${config.model_card_counts?' + model card':''}`,repoId:repositoryId,rawConfig:config,
      note:structural.length?`Architecture estimate detected ${structural.join(', ')}. Confirm exact checkpoint tensor counts before procurement.`:undefined,
      parameterMethod:published.total?(config.model_card_counts?'published model-card value':'explicit config value'):declared.total?(corroborated?'repository name corroborated by architecture':'repository-declared size'):'adaptive architecture estimate'
    };
  }

  function findRequiredGpuCount(maxGpu,evaluate){
    for(let n=1;n<=8;n++){
      const metrics=evaluate(n);
      if(metrics.meetsMemory&&metrics.meetsTps)return Object.assign({n,exceedsMax:n>maxGpu,unresolved:false},metrics);
    }
    let low=8,high=16,metrics=evaluate(high),doublings=0;
    while(!(metrics.meetsMemory&&metrics.meetsTps)&&doublings<48){low=high;high*=2;metrics=evaluate(high);doublings++}
    if(!(metrics.meetsMemory&&metrics.meetsTps))return Object.assign({n:high,exceedsMax:true,unresolved:true},metrics);
    while(low+1<high){
      const mid=Math.floor((low+high)/2),candidate=evaluate(mid);
      if(candidate.meetsMemory&&candidate.meetsTps){high=mid;metrics=candidate}else low=mid;
    }
    if(high!==metrics.n)metrics=evaluate(high);
    return Object.assign({n:high,exceedsMax:high>maxGpu,unresolved:false},metrics);
  }

  function normalizeUtilization(value,fallback){
    const defaultValue=fallback==null?.9:fallback;
    const parsed=Number(value);
    return Number.isFinite(parsed)&&parsed>0&&parsed<=1?parsed:defaultValue;
  }

  return {declaredCounts,languageConfig,parameterBillions,publishedCounts,modelCardCounts,estimateParams,cacheGeometry,normalizeConfig,findRequiredGpuCount,normalizeUtilization};
});

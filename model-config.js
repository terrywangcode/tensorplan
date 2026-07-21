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

  function estimateParams(config){
    const c=languageConfig(config);
    const h=c.hidden_size||c.d_model||4096;
    const L=c.num_hidden_layers||c.n_layer||32;
    const heads=c.num_attention_heads||c.n_head||32;
    const kvh=c.num_key_value_heads||c.num_kv_heads||heads;
    const hd=c.head_dim||Math.round(h/heads);
    const inter=c.intermediate_size||c.ffn_dim||h*4;
    const vocab=c.vocab_size||150000;
    const tie=c.tie_word_embeddings==null?(config.tie_word_embeddings!==false):c.tie_word_embeddings;
    const embeddingParams=vocab*h*(tie?1:2);
    const qDim=heads*hd,kvDim=kvh*hd;
    const fullAttention=h*(qDim+2*kvDim)+qDim*h+(c.attn_output_gate?h*qDim:0);
    const linearQ=(c.linear_num_key_heads||kvh)*(c.linear_key_head_dim||hd);
    const linearK=linearQ;
    const linearV=(c.linear_num_value_heads||heads)*(c.linear_value_head_dim||hd);
    const linearAttention=h*(linearQ+linearK+linearV+linearV)+linearV*h;
    const layerTypes=Array.isArray(c.layer_types)?c.layer_types:null;
    const fullLayers=layerTypes?layerTypes.filter(x=>String(x).toLowerCase()==='full_attention').length:L;
    const linearLayers=layerTypes?layerTypes.filter(x=>String(x).toLowerCase()==='linear_attention').length:0;
    const otherLayers=Math.max(0,L-fullLayers-linearLayers);
    const experts=c.num_experts||c.n_routed_experts||1;
    const topk=c.num_experts_per_tok||c.num_experts_per_token||c.num_experts_per_token||experts;
    const shared=c.n_shared_experts||0;
    const expertInter=c.moe_intermediate_size||inter;
    const mlpPerExpert=3*h*expertInter;
    const totalMlp=mlpPerExpert*(experts+shared);
    const activeMlp=mlpPerExpert*(Math.min(topk,experts)+shared);
    const attentionTotal=fullLayers*fullAttention+linearLayers*linearAttention+otherLayers*fullAttention;
    const mtpLayers=c.mtp_num_hidden_layers||0;
    const mtpParams=mtpLayers*(fullAttention+activeMlp);
    const total=embeddingParams+attentionTotal+L*totalMlp+mtpParams;
    const active=embeddingParams+attentionTotal+L*activeMlp+mtpParams;
    return {total:total/1e9,active:active/1e9,h,L,heads,kvh,hd,fullLayers,linearLayers,config:c};
  }

  function cacheGeometry(est){
    const c=est.config;
    let stateBytesPerSequence=0;
    if(est.linearLayers){
      const keyDim=c.linear_key_head_dim||est.hd;
      const valueDim=c.linear_value_head_dim||est.hd;
      const valueHeads=c.linear_num_value_heads||est.heads;
      const stateBytes=String(c.mamba_ssm_dtype||'float32').includes('32')?4:2;
      stateBytesPerSequence=est.linearLayers*valueHeads*keyDim*valueDim*stateBytes;
      const convChannels=(c.linear_num_key_heads||est.kvh)*keyDim*2+valueHeads*valueDim;
      stateBytesPerSequence+=est.linearLayers*convChannels*Math.max(0,(c.linear_conv_kernel_dim||1)-1)*2;
    }
    return {attentionFactor:est.fullLayers/est.L,stateBytesPerSequence};
  }

  function normalizeConfig(repositoryId,config,source){
    const est=estimateParams(config);
    const declared=declaredCounts(repositoryId);
    const hasExperts=(est.config.num_experts||est.config.n_routed_experts||1)>1;
    const params=config.num_parameters?config.num_parameters/1e9:(declared.total||est.total);
    const active=declared.active||(!hasExperts&&declared.total?declared.total:est.active);
    const cache=cacheGeometry(est);
    const corroborated=declared.total&&Math.abs(est.total-declared.total)/declared.total<.2;
    const cacheType=est.linearLayers?`Hybrid: ${est.fullLayers} full + ${est.linearLayers} linear`:(hasExperts?'MoE + GQA/MHA':'Config-derived GQA/MHA');
    return {
      id:`remote-${Date.now()}`,name:String(repositoryId).split('/').pop(),family:'Registry',
      params:+params.toFixed(2),active:+active.toFixed(2),layers:est.L,heads:est.heads,kvHeads:est.kvh,headDim:est.hd,
      maxContext:est.config.max_position_embeddings||est.config.seq_length||est.config.model_max_length||32768,
      attentionFactor:cache.attentionFactor,stateBytesPerSequence:cache.stateBytesPerSequence,cacheType,
      confidence:(config.num_parameters||corroborated)?'verified':'estimated',source:`${source} config.json`,repoId:repositoryId,rawConfig:config,
      note:est.linearLayers?'Linear-attention recurrent state is included as a per-request cache estimate. Confirm runtime allocation with the serving engine.':undefined,
      parameterMethod:config.num_parameters?'explicit config value':declared.total?(corroborated?'repository name corroborated by architecture':'repository-declared size'):'architecture estimate'
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

  return {declaredCounts,languageConfig,estimateParams,cacheGeometry,normalizeConfig,findRequiredGpuCount,normalizeUtilization};
});

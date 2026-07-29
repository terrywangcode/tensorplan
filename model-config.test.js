const assert=require('assert');
const {normalizeConfig,estimateParams,parameterBillions,modelCardCounts,findRequiredGpuCount,normalizeUtilization}=require('./model-config.js');

const qwen36={
  architectures:['Qwen3_5ForConditionalGeneration'],model_type:'qwen3_5',tie_word_embeddings:false,
  text_config:{hidden_size:5120,intermediate_size:17408,num_hidden_layers:64,num_attention_heads:24,num_key_value_heads:4,head_dim:256,vocab_size:248320,tie_word_embeddings:false,max_position_embeddings:262144,attn_output_gate:true,full_attention_interval:4,linear_num_key_heads:16,linear_num_value_heads:48,linear_key_head_dim:128,linear_value_head_dim:128,linear_conv_kernel_dim:4,mamba_ssm_dtype:'float32',mtp_num_hidden_layers:1,layer_types:Array.from({length:64},(_,i)=>(i+1)%4===0?'full_attention':'linear_attention')}
};

const estimate=estimateParams(qwen36);
assert.strictEqual(estimate.L,64,'must read nested text_config layers');
assert.strictEqual(estimate.h,5120,'must read nested text_config hidden size');
assert.ok(estimate.total>26&&estimate.total<29,`architecture estimate should be near 27B, got ${estimate.total}`);

const model=normalizeConfig('Qwen/Qwen3.6-27B',qwen36,'ModelScope');
assert.strictEqual(model.params,27);
assert.strictEqual(model.active,27);
assert.strictEqual(model.layers,64);
assert.strictEqual(model.cacheType,'Hybrid: 16 full + 48 linear');
assert.strictEqual(model.attentionFactor,.25);
assert.ok(model.stateBytesPerSequence>100*1024*1024,'linear-attention state cache must be included');
assert.strictEqual(model.confidence,'verified');

const kimiK3={
  architectures:['KimiK3ForConditionalGeneration'],model_type:'kimi_k3',tie_word_embeddings:false,
  text_config:{
    hidden_size:7168,intermediate_size:33792,num_hidden_layers:93,num_attention_heads:96,num_key_value_heads:96,
    vocab_size:163840,tie_word_embeddings:false,max_position_embeddings:1048576,first_k_dense_replace:1,moe_layer_freq:1,
    num_experts:896,num_experts_per_token:16,num_shared_experts:2,moe_intermediate_size:3072,
    routed_expert_hidden_size:3584,latent_moe_use_norm:true,q_lora_rank:1536,kv_lora_rank:512,
    qk_nope_head_dim:128,qk_rope_head_dim:64,v_head_dim:128,mla_use_output_gate:true,attn_res_block_size:12,
    linear_attn_config:{
      full_attn_layers:[...Array.from({length:23},(_,i)=>(i+1)*4),93],
      kda_layers:Array.from({length:91},(_,i)=>i+1).filter(i=>i%4!==0),
      head_dim:128,num_heads:96,short_conv_kernel_size:4,use_full_rank_gate:true
    }
  }
};

const kimiEstimate=estimateParams(kimiK3);
assert.strictEqual(kimiEstimate.moeLayers,92,'dense-prefix models must not apply MoE to every layer');
assert.strictEqual(kimiEstimate.denseLayers,1);
assert.strictEqual(kimiEstimate.expertInput,3584,'latent routed-expert width must override the transformer width');
assert.strictEqual(kimiEstimate.shared,2,'num_shared_experts must be recognized');
assert.strictEqual(kimiEstimate.fullLayers,24);
assert.strictEqual(kimiEstimate.linearLayers,69);
assert.ok(kimiEstimate.total>2750&&kimiEstimate.total<2820,`Kimi-K3 estimate should be near the published 2.8T, got ${kimiEstimate.total}`);
assert.ok(kimiEstimate.active>102&&kimiEstimate.active<107,`Kimi-K3 active estimate should be near the published 104B, got ${kimiEstimate.active}`);

const kimiModel=normalizeConfig('moonshotai/Kimi-K3',kimiK3,'ModelScope');
assert.strictEqual(kimiModel.params,2779.48);
assert.strictEqual(kimiModel.active,105.36);
assert.strictEqual(kimiModel.headDim,128);
assert.strictEqual(kimiModel.cacheType,'MoE + Hybrid: 24 full + 69 linear');
assert.strictEqual(kimiModel.parameterMethod,'adaptive architecture estimate');
assert.ok(kimiModel.stateBytesPerSequence>400*1024*1024,'KDA recurrent state must be represented in cache sizing');
assert.ok(kimiModel.note.includes('latent experts 3584-wide'));

const published=normalizeConfig('vendor/novel-model',{...kimiK3,total_parameters:'2.8T',activated_parameters:'104B'},'ModelScope');
assert.strictEqual(published.params,2800,'machine-readable published totals must take precedence over estimates');
assert.strictEqual(published.active,104,'machine-readable published active counts must take precedence over estimates');
assert.strictEqual(published.confidence,'verified');
assert.strictEqual(parameterBillions('2.8T'),2800);
assert.strictEqual(parameterBillions('104B'),104);

const cardCounts=modelCardCounts('<td><strong>Total Parameters</strong></td><td>2.8T</td><td>Activated Parameters</td><td>104B</td>');
assert.deepStrictEqual(cardCounts,{total_parameters:'2.8T',activated_parameters:'104B'});
const cardModel=normalizeConfig('moonshotai/Kimi-K3',{...kimiK3,model_card_counts:cardCounts},'ModelScope');
assert.strictEqual(cardModel.params,2800);
assert.strictEqual(cardModel.active,104);
assert.strictEqual(cardModel.parameterMethod,'published model-card value');

const overLimit=findRequiredGpuCount(64,n=>({meetsMemory:n>=73,meetsTps:true,capacity:n*24}));
assert.strictEqual(overLimit.n,73,'must report the actual requirement beyond the configured maximum');
assert.strictEqual(overLimit.exceedsMax,true);
assert.strictEqual(overLimit.unresolved,false);

const withinLimit=findRequiredGpuCount(64,n=>({meetsMemory:n>=3,meetsTps:true}));
assert.strictEqual(withinLimit.n,3,'must not round exact GPU demand to a power of two');
assert.strictEqual(withinLimit.exceedsMax,false);

const veryLarge=findRequiredGpuCount(64,n=>({meetsMemory:n>=20000,meetsTps:true}));
assert.strictEqual(veryLarge.n,20000,'must calculate large requirements without an arbitrary search cap');

assert.strictEqual(normalizeUtilization('0.90'),.9,'valid select value must be preserved');
assert.strictEqual(normalizeUtilization(''),.9,'blank select value must fall back instead of becoming zero');
assert.strictEqual(normalizeUtilization('0'),.9,'zero utilization must be rejected');
assert.strictEqual(normalizeUtilization('not-a-number',.85),.85,'invalid utilization must use the supplied fallback');

console.log(JSON.stringify({status:'pass',params:model.params,active:model.active,layers:model.layers,cache:model.cacheType,stateMiB:+(model.stateBytesPerSequence/1024/1024).toFixed(1),architectureEstimateB:+estimate.total.toFixed(2),kimiTotalB:+kimiEstimate.total.toFixed(2),kimiActiveB:+kimiEstimate.active.toFixed(2),overLimitGpuDemand:overLimit.n,exactNonPowerOfTwoDemand:withinLimit.n,veryLargeGpuDemand:veryLarge.n,resetUtilization:normalizeUtilization('')}));

const assert=require('assert');
const {normalizeConfig,estimateParams,findRequiredGpuCount,normalizeUtilization}=require('./model-config.js');

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

console.log(JSON.stringify({status:'pass',params:model.params,active:model.active,layers:model.layers,cache:model.cacheType,stateMiB:+(model.stateBytesPerSequence/1024/1024).toFixed(1),architectureEstimateB:+estimate.total.toFixed(2),overLimitGpuDemand:overLimit.n,exactNonPowerOfTwoDemand:withinLimit.n,veryLargeGpuDemand:veryLarge.n,resetUtilization:normalizeUtilization('')}));

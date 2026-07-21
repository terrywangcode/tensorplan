const assert=require('assert');
const fs=require('fs');

const html=fs.readFileSync('index.html','utf8');
const css=fs.readFileSync('styles.css','utf8');
const app=fs.readFileSync('app.js','utf8');
function captures(text,pattern){
  const values=[];
  let match;
  while((match=pattern.exec(text))!==null)values.push(match[1]);
  return values;
}

const ids=captures(html,/id="([^"]+)"/g);
const refs=captures(app,/\$\('([^']+)'\)/g);

assert.strictEqual(ids.length,new Set(ids).size,'HTML IDs must be unique');
assert.deepStrictEqual([...new Set(refs.filter(id=>!ids.includes(id)))],[],'Every JavaScript ID reference must exist');
assert.strictEqual(css.split('{').length,css.split('}').length,'CSS braces must balance');
assert.ok(html.indexOf('model-config.js')<html.indexOf('app.js'),'model-config.js must load before app.js');
for(const asset of ['styles.css','app.js','model-config.js','favicon.svg','site.webmanifest'])assert.ok(fs.existsSync(asset),`${asset} must exist`);
assert.ok(html.includes('https://terrywangcode.github.io/tensorplan/'),'Canonical public URL must be present');

console.log(JSON.stringify({status:'pass',ids:ids.length,jsReferences:new Set(refs).size,assets:5}));

// UI-only check: run against a dedicated CPU server; never submit model sampling.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');

(async () => {
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_PATH || undefined});
  try {
    const page = await browser.newPage({viewport:{width:1800,height:1200}});
    await page.goto(process.env.COMFY_URL || 'http://127.0.0.1:8191', {waitUntil:'networkidle'});
    await page.waitForFunction(() => !!window.app?.graph);
    for (const variant of ['Hybrid_Edit','Hybrid_NoCache_Edit','T2I','Spectrum','Sol','Edit']) {
      const file = path.join(root, 'workflows', `Qwen21_T8_1024_${variant}.json`);
      const saved = JSON.parse(await fs.readFile(file,'utf8'));
      await page.locator('#comfy-file-input').setInputFiles(file);
      await page.waitForFunction(() => window.app.graph._nodes.some(n => n.type === 'QwenImage21SpectrumT8'));
      const widgets = await page.evaluate(() => window.app.graph._nodes
        .filter(n => ['QwenImage21BlockCacheT8','QwenImage21SpectrumT8'].includes(n.type))
        .map(n => ({type:n.type, values:Object.fromEntries(n.widgets.map(w => [w.name,w.value]))})));
      for (const node of widgets) {
        const old = saved.nodes.find(n => n.type === node.type).widgets_values_named;
        for (const [name,value] of Object.entries(old)) assert.deepEqual(node.values[name],value,`${variant}/${name}`);
        assert.equal(node.values.threshold_mode,'constant');
        assert.equal(node.values.split_ratio,0.5);
        assert.equal(node.values.late_threshold,node.type==='QwenImage21BlockCacheT8'?0.03:0.08);
      }
      assert.equal(await page.evaluate(() => window.app.graph._nodes
        .find(n=>n.type==='QwenImage21SageAttentionT8').widgets.find(w=>w.name==='backend_mode').value),variant.startsWith('Hybrid')?'sage_kitchen':'sage');
      console.log('CANVAS_IMPORT_OK',variant);
    }
    const staged = await page.evaluate(async () => {
      window.app.graph._nodes.find(n=>n.type==='QwenImage21SageAttentionT8')
        .widgets.find(w=>w.name==='backend_mode').value='sage_kitchen';
      for (const n of window.app.graph._nodes) {
        if (!['QwenImage21BlockCacheT8','QwenImage21SpectrumT8'].includes(n.type)) continue;
        const values = {threshold_mode:'two_stage',start_percent:0.15,end_percent:0.85,split_ratio:0.5};
        values.late_threshold=n.type==='QwenImage21BlockCacheT8'?0.01:0.03;
        for (const [name,value] of Object.entries(values)) n.widgets.find(w=>w.name===name).value=value;
        n.size[1]=Math.max(n.size[1],n.computeSize()[1]);
      }
      return window.app.graph.serialize();
    });
    const out = path.join(root,'benchmark_results');
    await fs.mkdir(out,{recursive:true});
    const exported = path.join(out,'thresholds_canvas.json');
    await fs.writeFile(exported,JSON.stringify(staged,null,2));
    await page.evaluate(() => window.app.graph.clear());
    await page.locator('#comfy-file-input').setInputFiles(exported);
    await page.waitForFunction(() => window.app.graph._nodes.some(n=>n.type==='QwenImage21SpectrumT8'));
    const prompt = await page.evaluate(async () => (await window.app.graphToPrompt()).output);
    assert.equal(Object.values(prompt).find(n=>n.class_type==='QwenImage21SageAttentionT8').inputs.backend_mode,'sage_kitchen');
    for (const type of ['QwenImage21BlockCacheT8','QwenImage21SpectrumT8']) {
      const node = Object.values(prompt).find(n=>n.class_type===type);
      assert(node,type+' missing from prompt');
      assert.equal(node.inputs.threshold_mode,'two_stage');
      assert.equal(node.inputs.start_percent,0.15);
      assert.equal(node.inputs.end_percent,0.85);
      assert.equal(node.inputs.split_ratio,0.5);
      assert.equal(node.inputs.late_threshold,type==='QwenImage21BlockCacheT8'?0.01:0.03);
    }
    await page.screenshot({path:path.join(out,'thresholds_canvas.png')});
    console.log('TWO_STAGE_AND_HYBRID_IMPORT_AND_PROMPT_OK; no sampling submitted');
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});

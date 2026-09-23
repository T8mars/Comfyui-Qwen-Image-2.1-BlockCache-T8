// Opt-in browser test. Use an existing Playwright installation via PLAYWRIGHT_MODULE.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const variant = process.argv.includes('--sol') ? 'Sol' : process.argv.includes('--spectrum') ? 'Spectrum' : 'T2I';

(async () => {
  await fs.mkdir(path.join(root, 'benchmark_results'), {recursive: true});
  const lock = path.join(root, 'benchmark_results/run.lock');
  const lockFile = await fs.open(lock, 'wx');
  let browser;
  let pending = false;
  try {
  await lockFile.writeFile('Canvas test running. Do not run other GPU or API tests.\n');
  browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_PATH || undefined});
  const page = await browser.newPage({viewport: {width: 1800, height: 1200}});
  const executions = new Map();
  page.on('websocket', ws => ws.on('framereceived', ({payload}) => {
    if (typeof payload !== 'string') return;
    const event = JSON.parse(payload);
    if (event.type !== 'executing') return;
    const {prompt_id, node} = event.data;
    const state = executions.get(prompt_id) || {durations:{}};
    const now = performance.now();
    if (state.node != null) state.durations[state.node] = (now - state.started) / 1000;
    state.node = node;
    state.started = now;
    executions.set(prompt_id, state);
  }));
  page.on('pageerror', e => console.log('PAGEERROR', String(e)));
  await page.goto(process.env.COMFY_URL || 'http://127.0.0.1:8189', {waitUntil: 'networkidle'});
  await page.waitForFunction(() => !!window.app?.graph, {timeout: 60000});
  await fs.mkdir(path.join(root, 'benchmark_results'), {recursive: true});
  const workflow = await page.evaluate((variant) => {
    const app = window.app;
    const LG = window.LiteGraph;
    app.graph.clear();
    function create(type, title, position, values = {}, mode = 0) {
      const node = LG.createNode(type);
      if (!node) throw new Error('Unregistered node: ' + type);
      app.graph.add(node);
      node.title = title;
      node.pos = position;
      node.mode = mode;
      for (const [key, value] of Object.entries(values)) {
        const widget = node.widgets.find(w => w.name === key);
        if (!widget) throw new Error(type + ': unknown widget ' + key);
        widget.value = value;
      }
      node.properties['Node name for S&R'] = type;
      return node;
    }
    const model = create('UNETLoader', '01 / Official Qwen 2.1 model', [30,80], {
      unet_name:'qwen_image_2.1_int8_convrot.safetensors',weight_dtype:'default'});
    const clip = create('CLIPLoader', '02 / Qwen3-VL 8B text encoder', [30,300], {
      clip_name:'qwen3vl_8b_fp8_scaled.safetensors',type:'qwen_image'});
    const vae = create('VAELoader', '03 / Official Qwen 2.1 VAE', [30,520], {vae_name:'qwen_image_2.1_vae_bf16.safetensors'});
    const text = create('TextEncodeQwenImage21', '04 / Prompt + resolution / official', [430,350], {
      prompt:'A studio photograph of a small red ceramic teapot on a pale wooden table, a white card next to it clearly reads "QWEN 2.1", soft window lighting, a green plant in the background, realistic glaze and shadows, clean composition.',
      negative_prompt:'', resolution:1024});
    text.size = [370,350];
    clip.connect(0,text,text.findInputSlot('clip'));
    const kitchen = create('ModelAttentionBackend', '05 / Comfy Kitchen (official)', [430,80], {attention:'comfy kitchen attention'});
    const sage = create('QwenImage21SageAttentionT8', '06 / T8 Sage / BYPASSED', [850,80], {}, 4);
    const sol = create('QwenImage21SolAttentionT8', '07 / T8 Sol / BYPASSED at 1024', [850,240], {}, 4);
    const block = create('QwenImage21BlockCacheT8', '08 / T8 Block Cache / ON', [1260,80]);
    const spectrum = create('QwenImage21SpectrumT8', '09 / T8 Spectrum / BYPASSED', [1260,470], {}, 4);
    const sampler = create('KSampler', '10 / Native sampler / fixed seed', [1690,80], {
      seed:42,control_after_generate:'fixed',steps:25,cfg:1,sampler_name:'euler',scheduler:'simple',denoise:1});
    const decode = create('VAEDecode', '11 / Native VAE decode', [2100,80]);
    const save = create('SaveImage', '12 / Output (drag PNG back to restore)', [2400,80], {filename_prefix:'Qwen21_T8_Canvas'});
    save.size = [500,560];
    const chain = [model,kitchen,sage,sol,block,spectrum,sampler];
    chain.slice(0,-1).forEach((node,i) => node.connect(0,chain[i+1],chain[i+1].findInputSlot('model')));
    text.connect(0,sampler,sampler.findInputSlot('positive'));
    text.connect(1,sampler,sampler.findInputSlot('negative'));
    text.connect(2,sampler,sampler.findInputSlot('latent_image'));
    sampler.connect(0,decode,decode.findInputSlot('samples'));
    vae.connect(0,decode,decode.findInputSlot('vae'));
    decode.connect(0,save,save.findInputSlot('images'));
    const note = create('Note', 'READ ME / 使用说明', [1690,600], {
      text:'官方 Qwen-Image-2.1 文生图 / 1024 / 25步 / 固定seed42。\n默认启用 Kitchen + T8 Block Cache。\n紫色节点为旁路：选中后 Ctrl+B 启用/旁路。\nSage 与 Kitchen 选择一个；Sol、Spectrum 按需使用，不必全开。\n0.1.3 保留 Core 前缀 KV 缓存；Block/Spectrum 各用自己的连续上限。\n图像编辑请用 Qwen21_T8_1024_Edit.json，参考图和 VAE 都须接入文本编码节点。\nCPU cache只占缓存内存，不表示主模型在CPU执行。\n请自行选择已安装的模型，模型不自动下载。'});
    note.size = [470,330];
    text.pos = [30,740];
    kitchen.pos = [460,80];
    sage.pos = [460,240];
    sol.pos = [460,400];
    block.pos = [460,710];
    spectrum.pos = [870,80];
    sampler.pos = [870,500];
    decode.pos = [870,930];
    save.pos = [1260,80];
    note.pos = [1260,720];
    if (variant === 'Spectrum') {
      block.mode = 4;
      block.title = '08 / T8 Block Cache / BYPASSED';
      spectrum.mode = 0;
      spectrum.title = '09 / T8 Spectrum / ON';
      note.widgets[0].value = note.widgets[0].value.replace('默认启用 Kitchen + T8 Block Cache。','默认启用 Kitchen + T8 Spectrum，Block/Sage/Sol旁路。');
    }
    if (variant === 'Sol') {
      block.mode = 4;
      block.title = '08 / T8 Block Cache / BYPASSED';
      sol.mode = 0;
      sol.title = '07 / T8 Sol / ON / 1024 TEST';
      sol.widgets.find(w => w.name === 'enabled').value = true;
      sol.widgets.find(w => w.name === 'min_tokens').value = 4096;
      note.widgets[0].value = '1024 Sol独立实验 / 25步 / seed42 / tau1。\nKitchen + Sol启用；Block/Spectrum/Sage旁路。\nmin_tokens=4096、enabled=true；终端 kernel>0 才表示实际调用。\n已测安全余量：--reserve-vram 5 --vram-headroom 3 --disable-comfy-compiler。\n历史文生图约4%采样收益，但文字/细节变化；1MP编辑叠加Sol没有更快。\n严格串行，完成后卸载模型。2048曾发生系统重启，原因未明，暂停压力测试。\n关闭Sol：enabled=false 或 Ctrl+B旁路；编辑请用单独的 Edit 工作流。';
    }
    if (variant !== 'Sol') note.widgets[0].value += '\nSol的2048测试中发生过系统重启，默认disabled。测试必须严格串行，2048压力测试暂停。';
    save.widgets.find(w => w.name === 'filename_prefix').value = 'Qwen21_T8_Canvas_' + variant;
    app.canvas.ds.scale = 0.75;
    app.canvas.ds.offset = [100,80];
    return app.graph.serialize();
  }, variant);
  const output = path.join(root,'workflows',`Qwen21_T8_1024_${variant}.json`);
  await fs.mkdir(path.dirname(output), {recursive:true});
  await fs.writeFile(output, JSON.stringify(workflow, null, 2));
  // Import the serialized UI workflow through the frontend's real file input.
  await page.evaluate(() => window.app.graph.clear());
  await page.locator('#comfy-file-input').setInputFiles(output);
  await page.waitForFunction(() => window.app.graph._nodes.some(n => n.type === 'QwenImage21BlockCacheT8'));
  console.log('IMPORTED', await page.evaluate(() => window.app.graph._nodes.map(n => ({type:n.type,mode:n.mode}))));
  await page.getByRole('button', {name:/展开任务队列/}).click();
  await page.waitForTimeout(1000);
  await page.screenshot({path:path.join(root,`benchmark_results/canvas_${variant}_imported.png`)});
  if (process.argv.includes('--run')) {
    const queue = await page.evaluate(async () => (await (await fetch('/queue')).json()));
    if (queue.queue_running.length || queue.queue_pending.length) throw new Error('ComfyUI is busy; run tests strictly one at a time.');
    const responsePromise = page.waitForResponse(r => r.url().endsWith('/prompt') && r.request().method() === 'POST', {timeout:30000});
    responsePromise.catch(() => {});
    console.log('BUTTONS', await page.getByRole('button').allTextContents());
    pending = true;
    await page.getByRole('button',{name:'运行',exact:true}).click();
    const response = await responsePromise;
    const result = await response.json();
    if (response.status() !== 200) {
      pending = false;
      throw new Error(JSON.stringify(result));
    }
    if (!result.prompt_id) throw new Error(JSON.stringify(result));
    console.log('CANVAS_QUEUED', result.prompt_id);
    const queued = response.request().postDataJSON();
    await fs.writeFile(path.join(root,`benchmark_results/canvas_${variant}_queued.json`),JSON.stringify(queued,null,2));
    let history;
    for (let i=0;i<240;i++) {
      await page.waitForTimeout(1000);
      history = await page.evaluate(async id => (await (await fetch('/history/'+id)).json())[id],result.prompt_id);
      if (history) break;
    }
    if (['success', 'error'].includes(history?.status?.status_str)) pending = false;
    if (pending || !history.status.completed || history.status.status_str !== 'success') throw new Error(JSON.stringify(history));
    await fs.writeFile(path.join(root,`benchmark_results/canvas_${variant}_history.json`),JSON.stringify(history,null,2));
    const timings = executions.get(result.prompt_id)?.durations;
    if (!timings?.['10']) throw new Error('Sampler execution was not observed; not a valid canvas sampling test.');
    await fs.writeFile(path.join(root,`benchmark_results/canvas_${variant}_timings.json`),JSON.stringify(timings,null,2));
    await page.screenshot({path:path.join(root,`benchmark_results/canvas_${variant}_completed.png`)});
    console.log('CANVAS_SUCCESS',JSON.stringify(history.outputs),'SAMPLER_SECONDS',timings['10']);
  }
  } finally {
    try {
      if (browser) await browser.close();
    } finally {
      await lockFile.close();
      if (pending) console.error('Test may still be running. Lock retained:', lock, 'Verify the server is idle before removing it.');
      else await fs.unlink(lock);
    }
  }
})().catch(e => {console.error(e); process.exit(1)});

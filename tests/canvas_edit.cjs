// Serial native image-edit regression. Import, save, re-import and click Run on the canvas.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const {parseArgs} = require('node:util');
const {values: args} = parseArgs({options: {
  workflow: {type:'string'}, export: {type:'string'}, mode: {type:'string', default:'baseline'},
  url: {type:'string', default:'http://127.0.0.1:8190'},
  size: {type:'string', default:'1024'}, steps: {type:'string', default:'40'},
  guard: {type:'string', default:'0.25'}, threshold: {type:'string', default:'0.08'},
  backend: {type:'string', default:'sage'},
}});
const modes = ['baseline','block','spectrum','combined','sol','sol-block','sol-combined'];
if (!modes.includes(args.mode) || !args.workflow || ![512,1024].includes(+args.size) || !['sage','kitchen'].includes(args.backend)) throw Error('Invalid edit benchmark arguments');
const root = path.resolve(__dirname,'..');
const out = path.join(root,'benchmark_results');

(async () => {
  await fs.mkdir(out,{recursive:true});
  const lock = path.join(out,'run.lock');
  const handle = await fs.open(lock,'wx');
  let browser, pending = false;
  try {
    await handle.writeFile('Serial edit canvas test; do not submit other GPU work.\n');
    const stamp = new Date().toISOString().replace(/[:.]/g,'-');
    const name = `edit_${args.size}_${args.mode}_${stamp}`;
    browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_PATH || undefined});
    const page = await browser.newPage({viewport:{width:1800,height:1200}});
    const executions = new Map();
    page.on('websocket', ws => ws.on('framereceived',({payload}) => {
      if (typeof payload !== 'string') return;
      const event = JSON.parse(payload);
      if (event.type !== 'executing') return;
      const {prompt_id,node} = event.data;
      const state = executions.get(prompt_id) || {durations:{}};
      const now = performance.now();
      if (state.node != null) state.durations[state.node] = (now-state.started)/1000;
      state.node=node; state.started=now; executions.set(prompt_id,state);
    }));
    page.on('pageerror', e => console.log('PAGEERROR',String(e)));
    await page.goto(args.url,{waitUntil:'networkidle'});
    await page.waitForFunction(() => !!window.app?.graph,{timeout:60000});
    await page.locator('#comfy-file-input').setInputFiles(path.resolve(args.workflow));
    await page.waitForFunction(() => window.app.graph._nodes.some(n=>n.type==='QwenImage21BlockCacheT8'));
    const workflow = await page.evaluate(({args,name}) => {
      const app=window.app;
      const get=type=>app.graph._nodes.find(n=>n.type===type);
      const set=(node,key,value)=>{const w=node.widgets.find(w=>w.name===key); if(!w)throw Error(key); w.value=value;};
      const block=get('QwenImage21BlockCacheT8'), spectrum=get('QwenImage21SpectrumT8'), sol=get('QwenImage21SolAttentionT8');
      block.mode=args.mode.includes('block') || args.mode.includes('combined') ? 0 : 4;
      spectrum.mode=args.mode.includes('spectrum') || args.mode.includes('combined') ? 0 : 4;
      sol.mode=args.mode.includes('sol') ? 0 : 4;
      set(sol,'enabled',sol.mode===0); set(sol,'min_tokens',4096);
      set(block,'residual_diff_threshold',+args.threshold); set(spectrum,'guard_threshold',+args.guard);
      for(const n of [block,spectrum,sol]) n.title=n.type + (n.mode===0?' / ON':' / BYPASS');
      const sage=get('QwenImage21SageAttentionT8'), kitchen=get('ModelAttentionBackend');
      sage.mode=args.backend==='sage'?0:4; kitchen.mode=args.backend==='kitchen'?0:4;
      sage.title='T8 Sage'+(sage.mode===0?' / ON':' / BYPASS');
      kitchen.title='Official Kitchen'+(kitchen.mode===0?' / ON':' / BYPASS');
      const text=get('TextEncodeQwenImage21'), vae=get('VAELoader');
      vae.connect(0,text,text.findInputSlot('vae'));
      set(text,'resolution',+args.size);
      set(get('LoadImage'),'image','10A.jpg');
      set(get('ResolutionSelector'),'megapixels',(+args.size/1024)**2);
      const sampler=get('KSampler'); set(sampler,'seed',42); set(sampler,'control_after_generate','fixed'); set(sampler,'steps',+args.steps);
      set(get('SaveImage'),'filename_prefix',args.export?'Qwen21_T8_Edit':'qwen21_t8_edit/'+name);
      const note=get('Note');
      if(note) set(note,'text','Qwen 2.1 原生图像编辑 / Native image edit\n参考图和 VAE 都必须连接 TextEncodeQwenImage21。请上传自己的参考图。\nSeed 42 / '+args.steps+' steps / resolution '+args.size+'. Mode: '+args.mode+'; backend: '+args.backend+'.\nBlock threshold: '+args.threshold+'; Spectrum guard: '+args.guard+'.\nSage/Kitchen 选一个；紫色表示旁路，Ctrl+B切换。Sol本尺寸未测得额外收益。\n竖图尺寸由 ResolutionSelector 控制；保留参考图比例可改接编码器 LATENT 输出到采样器。\n保留官方 prefix KV 缓存。近似跳层仍会改变细节；请严格串行运行。');
      const layout = {
        UNETLoader:[30,60], CLIPLoader:[30,270], VAELoader:[30,470], LoadImage:[30,640],
        ModelAttentionBackend:[430,60], QwenImage21SageAttentionT8:[430,210],
        QwenImage21SolAttentionT8:[430,340], QwenImage21BlockCacheT8:[430,660],
        QwenImage21SpectrumT8:[830,60], TextEncodeQwenImage21:[830,460],
        ResolutionSelector:[830,940], EmptyLatentImage:[830,1190],
        KSampler:[1270,60], VAEDecode:[1270,500], SaveImage:[1670,60], Note:[1270,750],
      };
      for (const [type,pos] of Object.entries(layout)) get(type).pos=pos;
      get('LoadImage').size=[340,460]; text.size=[390,390];
      get('SaveImage').size=[460,640]; if(note)note.size=[520,400];
      app.canvas.ds.scale=0.65; app.canvas.ds.offset=[100,100];
      return app.graph.serialize();
    },{args,name});
    const workflowPath=args.export?path.resolve(args.export):path.join(out,name+'.json');
    await fs.writeFile(workflowPath,JSON.stringify(workflow,null,2));
    await page.evaluate(()=>window.app.graph.clear());
    await page.locator('#comfy-file-input').setInputFiles(workflowPath);
    await page.waitForFunction(()=>window.app.graph._nodes.some(n=>n.type==='KSampler'));
    for (const url of [args.url,'http://127.0.0.1:8189']) {
      const q=await (await fetch(url+'/queue')).json();
      if(q.queue_running.length || q.queue_pending.length)throw Error('Queue busy: '+url);
    }
    const responsePromise=page.waitForResponse(r=>r.url().endsWith('/prompt') && r.request().method()==='POST',{timeout:30000});
    responsePromise.catch(()=>{});
    pending=true;
    await page.getByRole('button',{name:/^(运行|Run)$/}).click();
    const response=await responsePromise;
    const result=await response.json();
    if(response.status()!==200){pending=false;throw Error(JSON.stringify(result));}
    if(!result.prompt_id)throw Error(JSON.stringify(result));
    console.log('QUEUED',name,result.prompt_id);
    await fs.writeFile(path.join(out,name+'_queued.json'),JSON.stringify(response.request().postDataJSON(),null,2));
    let history;
    for(let i=0;i<1200;i++){
      await page.waitForTimeout(1000);
      history=await page.evaluate(async id=>(await(await fetch('/history/'+id)).json())[id],result.prompt_id);
      if(history)break;
    }
    if(['success','error'].includes(history?.status?.status_str))pending=false;
    const timings=executions.get(result.prompt_id)?.durations;
    await fs.writeFile(path.join(out,name+'_result.json'),JSON.stringify({args,prompt_id:result.prompt_id,history,timings},null,2));
    if(pending || history.status.status_str!=='success')throw Error(JSON.stringify(history));
    if(!timings?.['10'])throw Error('Sampler was cached; invalid benchmark');
    await page.screenshot({path:path.join(out,name+'_canvas.png')});
    console.log('SUCCESS',args.mode,'sampler_seconds',timings['10'],'outputs',JSON.stringify(history.outputs));
    await fetch(args.url+'/free',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({unload_models:true,free_memory:true})});
  } finally {
    try {if(browser)await browser.close();}
    finally {await handle.close(); if(!pending)await fs.unlink(lock); else console.error('Pending execution; lock retained',lock);}
  }
})().catch(e=>{console.error(e);process.exitCode=1;});

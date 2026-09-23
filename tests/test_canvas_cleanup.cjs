// CPU-only lifecycle checks, not a substitute for real frontend validation.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const vm = require('node:vm');

(async () => {
  const source = await fs.readFile(path.join(__dirname, 'canvas_smoke.cjs'), 'utf8');
  for (const failure of ['navigation', 'submit', 'timeout', 'terminal_error']) {
    let browserClosed = false;
    let handleClosed = false;
    let unlocked = false;
    let evaluations = 0;
    let exitCode;
    const page = {
      on() {},
      async goto() { if (failure === 'navigation') throw new Error('navigation failed'); },
      async waitForFunction() {},
      async waitForTimeout() {},
      async screenshot() {},
      locator() { return {async setInputFiles() {}}; },
      getByRole(role, options) {
        return {
          async click() { if (options?.name === '运行' && failure === 'submit') throw new Error('submit outcome unknown'); },
          async allTextContents() { return []; },
        };
      },
      async waitForResponse() {
        return {status: () => 200, json: async () => ({prompt_id:'test'}), request: () => ({postDataJSON: () => ({})})};
      },
      async evaluate() {
        evaluations += 1;
        if (evaluations === 1) return {nodes:[], links:[]};
        if (evaluations === 4) return {queue_running:[], queue_pending:[]};
        if (evaluations >= 5 && failure === 'terminal_error') return {status:{completed:false,status_str:'error'}};
      },
    };
    const fakeFs = {
      async mkdir() {}, async writeFile() {},
      async open() { return {async writeFile() {}, async close() { handleClosed = true; }}; },
      async unlink() { unlocked = true; },
    };
    await vm.runInNewContext(source, {
      __dirname, performance,
      console: {log() {}, error() {}},
      process: {argv:['node','canvas_smoke.cjs','--run'],env:{},exit(code) { exitCode = code; }},
      require(name) {
        if (name === 'playwright') return {chromium:{async launch() {
          return {async newPage() { return page; }, async close() { browserClosed = true; }};
        }}};
        if (name === 'node:fs/promises') return fakeFs;
        if (name === 'node:path') return path;
        throw new Error('Unexpected dependency: ' + name);
      },
    });
    assert.equal(exitCode, 1, failure);
    assert.equal(browserClosed, true, failure);
    assert.equal(handleClosed, true, failure);
    assert.equal(unlocked, failure === 'navigation' || failure === 'terminal_error', failure);
    console.log('PASS', failure);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

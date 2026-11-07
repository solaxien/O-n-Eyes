// Run a project-owned diagnostic expression in its existing Chrome tab over ADB.
const fs = require('node:fs');
const [origin, expressionFile, output, port = '49228',tabId] = process.argv.slice(2);
(async () => {
  const tabs = await fetch(`http://127.0.0.1:${port}/json/list`).then(r => r.json());
  const target = tabs.find(t => t.type === 'page' && (!tabId||t.id===tabId) && new URL(t.url).origin === new URL(origin).origin);
  if (!target) throw Error('Project tab not found');
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.addEventListener('open', resolve, {once:true});
    ws.addEventListener('error', reject, {once:true});
  });
  const value = await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(Error('Browser diagnostic timed out')), 600000);
    ws.addEventListener('close',()=>{clearTimeout(timeout);reject(Error('Project DevTools connection closed'));},{once:true});
    ws.addEventListener('message', e => {
      const message = JSON.parse(e.data);
      if (message.id !== 1) return;
      clearTimeout(timeout);
      if (message.error || message.result.exceptionDetails) reject(Error(JSON.stringify(message.error || message.result.exceptionDetails)));
      else resolve(message.result.result.value);
    });
    ws.send(JSON.stringify({id:0,method:'Page.bringToFront'}));
    ws.send(JSON.stringify({id:1,method:'Runtime.evaluate',params:{expression:fs.readFileSync(expressionFile,'utf8'),awaitPromise:true,returnByValue:true}}));
  });
  ws.close();
  fs.writeFileSync(output,JSON.stringify(value,null,2)+'\n');
  console.log(JSON.stringify(value?.metadata?{metadata:value.metadata,artifacts:Object.keys(value).filter(k=>k!=='metadata')}:value,null,2));
})().catch(e=>{console.error(e);process.exit(1);});

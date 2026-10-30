const {chromium}=require('playwright');
(async()=>{const b=await chromium.launch();
for(const f of process.argv.slice(2)){const [file,w,h]=f.split(':');
const p=await b.newPage({viewport:{width:+w,height:+h},deviceScaleFactor:3});
await p.goto('file://'+process.cwd()+'/'+file);await p.waitForTimeout(300);
await p.screenshot({path:file.replace('.html','.png')});await p.close();}
await b.close();})();

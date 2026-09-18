// frontend_assets.cpp — eingebettete Nova-4-Web-UI (Claude/ChatGPT-Stil).
//
// Single-File-SPA (kein externes Asset, kein CDN): Sidebar mit Chat-Liste +
// Einstellungen, mehrere Chats, Token-Streaming mit Markdown/Code-Rendering,
// Datei-Upload/Download, Skill-Installation. Spricht das WS-Protokoll aus
// chat_service (message/new_chat/list_chats/switch_chat/delete_chat/rename_chat)
// und die HTTP-Routen (/api/config, /api/persona, /upload, /download, /skill).
#include "Web/frontend_assets.h"

namespace nova::web {

const char* index_html() {
    return R"HTML(<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nova 4</title>
<style>
  :root{ color-scheme:dark; --bg:#0e0f13; --panel:#15171d; --panel2:#1b1d24; --line:#23262e;
         --txt:#e6e6e6; --dim:#8b93a7; --accent:#5b8cff; --accent2:#1f6feb33; }
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,Segoe UI,sans-serif;background:var(--bg);color:var(--txt);
       height:100vh;display:flex;overflow:hidden}
  /* Sidebar */
  #side{width:270px;background:var(--panel);border-right:1px solid var(--line);display:flex;
        flex-direction:column;transition:margin .2s}
  #side h1{font-size:15px;margin:0;padding:14px 16px;border-bottom:1px solid var(--line);
           display:flex;align-items:center;gap:8px}
  #side h1 .dot{width:8px;height:8px;border-radius:50%;background:#3fb950}
  #newchat{margin:10px;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--panel2);
           color:var(--txt);cursor:pointer;font-size:14px}
  #newchat:hover{border-color:var(--accent)}
  #chats{flex:1;overflow-y:auto;padding:4px 8px}
  .chat{padding:9px 10px;border-radius:8px;cursor:pointer;font-size:13.5px;color:var(--dim);
        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:flex;justify-content:space-between;gap:6px}
  .chat:hover{background:var(--panel2);color:var(--txt)}
  .chat.act{background:var(--accent2);color:var(--txt)}
  .chat .x{opacity:0;color:var(--dim)}
  .chat:hover .x{opacity:.7}
  #side footer{padding:10px;border-top:1px solid var(--line);display:flex;gap:8px}
  #side footer button{flex:1;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--panel2);
                      color:var(--dim);cursor:pointer;font-size:13px}
  #side footer button:hover{color:var(--txt);border-color:var(--accent)}
  /* Main */
  #main{flex:1;display:flex;flex-direction:column;min-width:0}
  #top{padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel);font-size:14px;
       display:flex;align-items:center;gap:10px}
  #burger{display:none;background:none;border:0;color:var(--txt);font-size:20px;cursor:pointer}
  #st{color:var(--dim);font-size:12px}
  #log{flex:1;overflow-y:auto;padding:24px 0}
  .row{max-width:820px;margin:0 auto;padding:6px 20px}
  .msg{padding:12px 16px;border-radius:12px;line-height:1.55;white-space:normal;word-wrap:break-word}
  .user .msg{background:var(--accent2);margin-left:auto;max-width:80%;width:fit-content}
  .nova .msg{background:var(--panel2)}
  .card{background:#2a2410;border:1px solid #6b531f;border-radius:10px;padding:8px 12px;font-family:ui-monospace,monospace;
        font-size:12.5px;color:#e8c877;margin:4px 0}
  .msg pre{background:#0a0b0e;border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto;position:relative}
  .msg pre code{font-family:ui-monospace,Consolas,monospace;font-size:13px}
  .msg code{background:#0a0b0e;padding:1px 5px;border-radius:4px;font-family:ui-monospace,monospace;font-size:13px}
  .copy{position:absolute;top:6px;right:6px;font-size:11px;background:var(--panel);border:1px solid var(--line);
        color:var(--dim);border-radius:5px;padding:2px 7px;cursor:pointer}
  /* Input */
  #bar{border-top:1px solid var(--line);background:var(--panel);padding:12px}
  #barin{max-width:820px;margin:0 auto;display:flex;gap:8px;align-items:flex-end}
  #in{flex:1;padding:11px 12px;border-radius:10px;border:1px solid var(--line);background:var(--bg);
      color:var(--txt);font-size:15px;resize:none;max-height:180px;font-family:inherit}
  #bar button{padding:11px 14px;border:0;border-radius:10px;background:var(--accent);color:#fff;cursor:pointer;font-size:15px}
  #attach{background:var(--panel2);border:1px solid var(--line);color:var(--dim)}
  /* Modal */
  #modal{position:fixed;inset:0;background:#000a;display:none;align-items:center;justify-content:center;z-index:5}
  #modal .box{background:var(--panel);border:1px solid var(--line);border-radius:12px;width:min(720px,92vw);
              max-height:88vh;overflow:auto;padding:18px}
  #modal h2{margin:.2em 0 .6em;font-size:16px}
  #modal textarea{width:100%;height:220px;background:var(--bg);color:var(--txt);border:1px solid var(--line);
                  border-radius:8px;padding:10px;font-family:ui-monospace,monospace;font-size:13px}
  #modal .actions{display:flex;gap:8px;justify-content:flex-end;margin-top:12px}
  #modal button{padding:9px 14px;border:1px solid var(--line);border-radius:8px;background:var(--panel2);
                color:var(--txt);cursor:pointer}
  #modal .save{background:var(--accent);border-color:var(--accent);color:#fff}
  .hint{color:var(--dim);font-size:12px;margin:4px 0 12px}
  @media(max-width:760px){ #side{position:absolute;height:100%;z-index:4;margin-left:-270px}
                           #side.open{margin-left:0} #burger{display:block} .user .msg{max-width:92%} }
</style>
</head>
<body>
<aside id="side">
  <h1><span class="dot"></span> Nova 4</h1>
  <button id="newchat">+ Neuer Chat</button>
  <div id="chats"></div>
  <footer>
    <button onclick="openSettings()">Einstellungen</button>
    <button onclick="openSkill()">+ Skill</button>
  </footer>
</aside>
<main id="main">
  <div id="top"><button id="burger" onclick="document.getElementById('side').classList.toggle('open')">&#9776;</button>
       <span id="title">Nova 4</span><span id="st">&mdash; verbinde&hellip;</span></div>
  <div id="log"></div>
  <div id="bar"><div id="barin">
    <button id="attach" title="Datei hochladen" onclick="document.getElementById('file').click()">&#128206;</button>
    <input type="file" id="file" style="display:none">
    <textarea id="in" rows="1" placeholder="Nachricht an Nova&hellip;"></textarea>
    <button onclick="send()">Senden</button>
  </div></div>
</main>
<div id="modal"><div class="box" id="modalbox"></div></div>
<script>
const $=s=>document.querySelector(s), log=$("#log");
let ws, active=null, cur=null, curRaw="", streaming=false;

function esc(t){return t.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function md(t){ // minimaler Markdown-Renderer
  let h=esc(t);
  h=h.replace(/```(\w*)\n?([\s\S]*?)```/g,(m,l,c)=>'<pre><button class="copy" onclick="cp(this)">copy</button><code>'+c.replace(/</g,'&lt;')+'</code></pre>');
  h=h.replace(/`([^`\n]+)`/g,'<code>$1</code>');
  h=h.replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>');
  h=h.replace(/^### (.*)$/gm,'<h3>$1</h3>').replace(/^## (.*)$/gm,'<h3>$1</h3>');
  h=h.replace(/\n/g,'<br>');
  return h;
}
window.cp=b=>{navigator.clipboard.writeText(b.parentElement.innerText.replace(/^copy/,''));b.textContent='ok';setTimeout(()=>b.textContent='copy',900);};
function addRow(cls){const r=document.createElement('div');r.className='row '+cls;const m=document.createElement('div');m.className='msg';r.appendChild(m);log.appendChild(r);log.scrollTop=log.scrollHeight;return m;}
function addCard(txt){const r=document.createElement('div');r.className='row nova';const c=document.createElement('div');c.className='card';c.textContent=txt;r.appendChild(c);log.appendChild(r);log.scrollTop=log.scrollHeight;}

function connect(){
  ws=new WebSocket('ws://'+location.host+'/ws');
  ws.onopen=()=>{ $("#st").textContent='&mdash; verbunden'; ws.send(JSON.stringify({type:'list_chats'})); };
  ws.onclose=()=>{ $("#st").textContent='&mdash; getrennt'; setTimeout(connect,1000); };
  ws.onmessage=e=>{ let m; try{m=JSON.parse(e.data)}catch{return;}
    if(m.type==='text'){ if(!cur){cur=addRow('nova');curRaw='';} curRaw+=m.text; cur.innerHTML=md(curRaw); log.scrollTop=log.scrollHeight; }
    else if(m.type==='tool'){ addCard('\u{1F527} tool_call: '+m.name); }
    else if(m.type==='apex'){ addCard('⚡ apex: '+m.skill); }
    else if(m.type==='done'){ cur=null; streaming=false; }
    else if(m.type==='chats'){ renderChats(m.chats); }
    else if(m.type==='chat_created'){ active=m.chat; setTitle(m.title); }
    else if(m.type==='history_turn'){ const mm=addRow(m.role==='user'?'user':'nova'); mm.innerHTML=md(m.text); }
    else if(m.type==='history_done'){ log.scrollTop=log.scrollHeight; }
  };
}
function renderChats(list){
  const c=$("#chats"); c.innerHTML='';
  list.forEach(ch=>{ const d=document.createElement('div'); d.className='chat'+(ch.id===active?' act':'');
    const t=document.createElement('span'); t.textContent=ch.title||'Chat'; t.style.overflow='hidden'; t.style.textOverflow='ellipsis';
    const x=document.createElement('span'); x.className='x'; x.textContent='✕';
    x.onclick=ev=>{ev.stopPropagation(); ws.send(JSON.stringify({type:'delete_chat',chat:ch.id})); if(ch.id===active){active=null;log.innerHTML='';setTitle('Nova 4');}};
    d.onclick=()=>switchChat(ch.id,ch.title);
    d.appendChild(t); d.appendChild(x); c.appendChild(d); });
}
function switchChat(id,title){ active=id; setTitle(title); log.innerHTML=''; cur=null;
  ws.send(JSON.stringify({type:'switch_chat',chat:id})); document.getElementById('side').classList.remove('open');
  ws.send(JSON.stringify({type:'list_chats'})); }
function setTitle(t){ $("#title").textContent=t||'Nova 4'; }
$("#newchat").onclick=()=>{ active=null; log.innerHTML=''; cur=null; setTitle('Neuer Chat'); ws.send(JSON.stringify({type:'new_chat'})); };

function send(){
  const i=$("#in"); const t=i.value.trim(); if(!t||streaming) return;
  const mm=addRow('user'); mm.textContent=t; i.value=''; i.style.height='auto'; cur=null; streaming=true;
  ws.send(JSON.stringify({type:'message',chat:active||'',text:t}));
  setTimeout(()=>ws.send(JSON.stringify({type:'list_chats'})),300);
}
$("#in").addEventListener('input',e=>{e.target.style.height='auto';e.target.style.height=Math.min(180,e.target.scrollHeight)+'px';});
$("#in").addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();} });

// Datei-Upload -> /upload?path=workspace\uploads\<name>, dann Nachricht mit dem Pfad
$("#file").addEventListener('change',async e=>{
  const f=e.target.files[0]; if(!f) return;
  const dest='uploads\\'+f.name;
  const r=await fetch('/upload?path='+encodeURIComponent(dest),{method:'POST',body:f});
  const j=await r.json().catch(()=>({ok:false}));
  addCard(j.ok?('\u{1F4C4} hochgeladen: '+dest):'Upload fehlgeschlagen');
  if(j.ok){ const note='Ich habe die Datei "'+dest+'" hochgeladen.'; const mm=addRow('user'); mm.textContent=note;
    streaming=true; cur=null; ws.send(JSON.stringify({type:'message',chat:active||'',text:note})); }
  e.target.value='';
});

// ----- Einstellungen -----
async function openSettings(){
  const cfg=await fetch('/api/config').then(r=>r.text()).catch(()=>'{}');
  const persona=await fetch('/api/persona').then(r=>r.text()).catch(()=>'');
  $("#modalbox").innerHTML=`<h2>Einstellungen</h2>
    <div class="hint">config.json &mdash; Neustart uebernimmt Aenderungen.</div>
    <textarea id="cfg"></textarea>
    <h2 style="margin-top:16px">persona.md</h2>
    <textarea id="persona"></textarea>
    <div class="actions"><button onclick="closeModal()">Abbrechen</button>
      <button class="save" onclick="saveSettings()">Speichern</button></div>`;
  $("#cfg").value=cfg; $("#persona").value=persona; showModal();
}
async function saveSettings(){
  await fetch('/api/config',{method:'POST',body:$("#cfg").value});
  await fetch('/upload?path='+encodeURIComponent('..\\memory\\persona.md'),{method:'POST',body:$("#persona").value});
  closeModal(); addCard('Einstellungen gespeichert (Neustart uebernimmt sie).');
}
// ----- Skill installieren -----
function openSkill(){
  $("#modalbox").innerHTML=`<h2>Skill installieren</h2>
    <div class="hint">YAML (name/type/tier/description/trigger_keywords/executor). Neustart laedt ihn.</div>
    <input id="skname" placeholder="skill-name" style="width:100%;padding:9px;background:var(--bg);color:var(--txt);border:1px solid var(--line);border-radius:8px;margin-bottom:8px">
    <textarea id="skyaml" placeholder="name: mein_tool&#10;type: tool&#10;tier: 1&#10;description: ..."></textarea>
    <div class="actions"><button onclick="closeModal()">Abbrechen</button>
      <button class="save" onclick="saveSkill()">Installieren</button></div>`;
  showModal();
}
async function saveSkill(){
  const n=$("#skname").value.trim(); if(!n) return;
  const r=await fetch('/skill?name='+encodeURIComponent(n),{method:'POST',body:$("#skyaml").value});
  const j=await r.json().catch(()=>({ok:false}));
  closeModal(); addCard(j.ok?('Skill "'+n+'" installiert (Neustart laedt ihn).'):'Skill-Install fehlgeschlagen');
}
function showModal(){ $("#modal").style.display='flex'; }
window.closeModal=()=>{ $("#modal").style.display='none'; };
$("#modal").addEventListener('click',e=>{ if(e.target.id==='modal') closeModal(); });

connect();
</script>
</body>
</html>
)HTML";
}

const char* mobile_html() { return index_html(); }  // SPA ist responsive

}  // namespace nova::web

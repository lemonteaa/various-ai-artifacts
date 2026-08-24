I think the sample ivr webpage used an outdated API. Please refer to the doc and help me update it. **show only the js code part.**

```js
(function () {
  const $ = id => document.getElementById(id);

  // The page is served by Asterisk itself, so location.hostname is already
  // the correct address of the PBX from the browser's point of view.
  const host = location.hostname;
  $("wsUrl").value  = "wss://" + host + ":8089/ws";
  $("domain").value = host;

  let simpleUser = null;
  let callState = "idle";           // idle | calling | in-call

  function setStatus(msg, cls) {
    $("status").textContent = msg;
    $("status").className = cls || "";
  }
  function log(msg) {
    const el = $("log");
    el.textContent += new Date().toLocaleTimeString() + "  " + msg + "\n";
    el.scrollTop = el.scrollHeight;
  }
  function digits() { return $("display").textContent.trim(); }
  function setDigits(d) { $("display").textContent = d || "\u00a0"; }
  function setCallState(s) {
    callState = s;
    $("btnCall").disabled = !(s === "idle" && digits());
    $("btnHang").disabled = !(s === "calling" || s === "in-call");
  }

  // ---------- keypad ----------
  const keys = [["1",""],  ["2","ABC"],["3","DEF"],
                ["4","GHI"],["5","JKL"],["6","MNO"],
                ["7","PQRS"],["8","TUV"],["9","WXYZ"],
                ["*",""],  ["0","+"],  ["#",""]];
  for (const [d, sub] of keys) {
    const b = document.createElement("button");
    b.innerHTML = d + (sub ? "<small>" + sub + "</small>" : "");
    b.onclick = () => onKey(d);
    $("keypad").appendChild(b);
  }
  // Backspace row
  const bs = document.createElement("button");
  bs.textContent = "⌫"; bs.style.gridColumn = "2";
  bs.onclick = () => { setDigits(digits().slice(0, -1)); setCallState(callState); };
  $("keypad").appendChild(bs);

  function onKey(d) {
    if (callState === "in-call" || callState === "calling") {
      // In-call: send DTMF immediately (this is what drives the IVR menus)
      try { simpleUser.sendDTMF(d); log("DTMF sent: " + d); }
      catch (e) { log("DTMF failed: " + e); }
    } else {
      setDigits(digits() + d);
      setCallState(callState);
    }
  }

  // ---------- SIP.js (SimpleUser) ----------
  $("btnConnect").onclick = async () => {
    try {
      setStatus("connecting…");

      // Build AOR (SIP URI) and server WebSocket from form
      const user   = $("user").value;
      const pass   = $("pass").value;
      const domain = $("domain").value;
      const wsUrl  = $("wsUrl").value;

      const server = wsUrl;
      const options = {
        aor: `sip:${user}@${domain}`,
        media: {
          constraints: { audio: true, video: false }
        },
        userAgentOptions: {
          authorizationUsername: user,
          authorizationPassword: pass,
          displayName: "Browser " + user
        }
      };

      simpleUser = new SIP.Web.SimpleUser(server, options);

      // Delegate events
      simpleUser.delegate = {
        onCallReceived: async () => {
          log("incoming call — auto-answering");
          setCallState("in-call");
          await simpleUser.answer();
          setStatus("in call (incoming)");
        },
        onCallHangup: () => {
          log("remote end hung up");
          setDigits(""); setCallState("idle"); setStatus("call ended");
        },
        onServerDisconnect: () => {
          log("server disconnected");
          setDigits(""); setCallState("idle");
          setStatus("disconnected from server", "err");
        }
      };

      await simpleUser.connect();
      log("WebSocket connected");

      await simpleUser.register();
      setStatus(`registered as ${user} — dial an extension`, "ok");
      setCallState("idle");
      $("btnConnect").disabled = true;
    } catch (e) {
      setStatus("connect failed: " + e, "err");
      log(String(e));
    }
  };

  $("btnCall").onclick = async () => {
    const target = digits();
    if (!target || !simpleUser) return;
    try {
      setCallState("calling");
      setStatus("calling " + target + " … (allow mic access if prompted)");
      const dest = `sip:${target}@${$("domain").value}`;
      log("calling " + dest);
      await simpleUser.call(dest);  // media constraints are already set in constructor
      setCallState("in-call");
      setStatus("in call with " + target + " — keypad now sends DTMF", "ok");
    } catch (e) {
      setStatus("call failed: " + e, "err");
      log(String(e));
      setCallState("idle");
    }
  };

  $("btnHang").onclick = async () => {
    try { await simpleUser.hangup(); }
    catch (e) { log(String(e)); }
    setDigits(""); setCallState("idle"); setStatus("call ended");
  };
})();
```

ok quick followup: I'm doing a asterisk tutorial. using the webpage above after fix, I can connect and dail. from asterisk server side, it shows the IVR ran successfully, and rtp debug log server side shows that RTP packet type 0 are **sent and received**. On client side, I'm using firefox, about:webrtc seems to show things are normal, yet I did NOT hear any audio (the IVR should playback asterisk standard audio to the client, then hang up). the hang up from IVR program does work. any suggestion?

The issue is likely that the remote audio stream is not being attached to an audio element. Your HTML doesn't include any `<audio>` or `<video>` element, so even though RTP packets are received, the browser has no destination to play the audio.

## Fix:
1. Add an `<audio>` element to your HTML:
   ```html
   <audio id="remoteAudio" autoplay></audio>
   ```

2. Modify your JavaScript to either pass that element to `SimpleUser` options, or attach the remote stream after the call is established.

### Option A: Pass it in constructor
In your `$("btnConnect").onclick` handler, change the `options` object to include:
```js
const options = {
  aor: `sip:${user}@${domain}`,
  media: {
    constraints: { audio: true, video: false },
    remote: {
      audio: document.getElementById('remoteAudio')
    }
  },
  userAgentOptions: { ... }
};
```

### Option B: Attach after call
After `await simpleUser.call(...)` succeeds (or in the `onCallReceived` delegate), add:
```js
const remoteAudio = document.getElementById('remoteAudio');
remoteAudio.srcObject = simpleUser.remoteMediaStream;
```

This ensures that the incoming audio stream is routed to the audio element.

## Other checks:
- Ensure the browser tab is not muted and volume is up.
- Confirm that the `remoteMediaStream` contains audio tracks (you can log `simpleUser.remoteMediaStream.getAudioTracks().length`).

If you still don't hear audio, check that the server is sending the correct codec (e.g., PCMU) and that Firefox supports it. By default, Firefox uses Opus, but Asterisk might be configured for PCMU; SIP.js should negotiate, but you can verify in the WebRTC stats.

Let me know if you need the full corrected JS.

----

(Human note: add `setDigits(digits() + d);` when want to update numpad UI for the new digit)

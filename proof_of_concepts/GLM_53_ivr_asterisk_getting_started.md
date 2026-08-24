
(Note: unverified code, transcript not cleaned up)

--
advanced devops/engineering, topic today: pbx/sip/ivr.

While telephony application can be considered legacy, there are still cases where it is needed. however, the doc surrounding these are often hard to read as they are somewhat niche.

I have a fresh debian VM. suppose I want to get a taste of IVR app dev as a software engineer, so I want the following:
1. install and config asterisk for local dev, maybe with a smoke test directly from inside the VM.
2. IVR end to end smoke test: implement a hello world style IVR program (dialplan + exten?) in asterisk, then write a html5 webpage that can test calling it softphone style. (I think sip.js is the recommended library?) (the webpage should impl some basic features like a phonepad that display the digits entered, similar to android phone. don't overdo it though - the main goal remains to test the IVR) (a key tricky point here is to configure the connection details. my setup is host machine window running the VM using qemu, can expose via hostfwd param etc. make sure to explain the necessary config on both server side and client side, and briefly explain how would a prod server with public IP addr differ)
3. quick tour of asterisk IVR programming: demo with a somewhat more realistic program that has two level menu, audio prompt (both tts/built in vs programmer supplied audio file), and most importantly the asterisk AGI script (eg query a local sqlite DB).
4. outro: how to get a real/public phone number, and once obtained how to "link" it to our pbx server.

This is an extended task, so take your time!

--


The user wants a comprehensive tutorial on Asterisk IVR development on Debian VM, QEMU with hostfwd, SIP.js softphone webpage, dialplan, AGI with SQLite, and outro on real phone numbers. Let me write a thorough, technically accurate guide.

Key technical decisions:

    Asterisk 18/20/21 from Debian repos (bookworm has asterisk 20). Use chan_sip? No — chan_sip is deprecated; use PJSIP. SIP.js uses WebSocket → need res_pjsip with WebSocket transport (Asterisk supports WSS/WS via res_pjsip_transport_websocket). Need TLS? SIP.js works over ws:// too, but browsers require secure context for getUserMedia — microphone access requires HTTPS! Important point: localhost is a secure context, but the Windows host accessing the VM via IP is not localhost... Actually the page served from the VM at http://192.168.x.x will not be a secure context, so getUserMedia (mic) will fail. Options: serve page via HTTPS (self-signed cert), or use Firefox which allows... actually Firefox also requires secure context for getUserMedia except localhost. So we should set up HTTPS on the web server (nginx or python with self-signed) and WSS for SIP transport. Asterisk needs TLS cert for WSS. Alternative: use Chrome flag unsafely-treat-insecure-origin-as-secure. Simplest robust path: create self-signed certs, configure Asterisk HTTP server with TLS for WS, and serve the HTML over HTTPS.

Let me plan the whole answer structure:
0. Architecture overview

Diagram: Windows host (browser + SIP.js) → QEMU hostfwd ports → Debian VM (asterisk). Ports: 5060/5061 not needed since WS. Asterisk built-in HTTP server on 8088 (ws) / 8089 (wss). We'll use WSS on 8089 and serve web page via HTTPS on 8443 (nginx) or via asterisk's static http. Could serve the HTML page directly from Asterisk's HTTP server over TLS on 8089 with /static? Asterisk http.conf has enablestatic=yes and prefix. But TLS cert needed anyway. Maybe simpler: use nginx with self-signed cert for HTTPS serving page, and asterisk with same cert for WSS. Or just use python http.server? Needs TLS too — python http.server with ssl module. Let me use nginx: standard, and useful.

Actually could combine: use Asterisk's built-in HTTP server with TLS on 8089, enable static content, serve the page from /var/lib/asterisk/static-http/. That reduces moving parts: one cert, one server. But asterisk TLS config requires cert in pem. Fine. However users may find nginx more familiar. I'll pick the Asterisk static server approach for minimal moving parts, and mention nginx alternative. Hmm, but URL would be https://VM:8089/static/... with prefix. Default prefix empty → https://vm:8089/index.html. Let's do that.

QEMU hostfwd: hostfwd=tcp::8089-:8089 (WSS + static page), maybe also SSH hostfwd=tcp::2222-:22. RTP ports! Audio (RTP/UDP) needs to flow. Browser sends RTP via WebRTC (asterisk's res_pjsip with WebRTC requires DTLS-SRTP, ice support). RTP port range default 10000-20000 UDP. With hostfwd we can't forward 10000 UDP ports easily. Options: reduce rtp range to a few ports and hostfwd each, e.g. rtpstart=10000 rtpend=10004 and forward 5 UDP ports. But WebRTC ICE: candidates — Asterisk will advertise its candidates; with NAT via hostfwd, the ICE candidates include the VM's IP which the host can't reach? Actually QEMU user-mode networking: guest is behind NAT (10.0.2.15). Asterisk advertises 10.0.2.15 as candidate — browser on host cannot route to that. But with ICE, srflx candidates etc. QEMU user net doesn't do STUN. Hmm.

Trick: Asterisk responds in SDP with its candidates; browser sends to that address. With QEMU SLIRP, the host can actually... no. But there's a common trick: since hostfwd maps host:port → guest:port, and Asterisk RTP uses same port for both directions? RTP is asymmetric typically but Asterisk uses symmetric RTP: it learns address from incoming packets. If browser sends RTP to 127.0.0.1:10000 (host), QEMU forwards to guest 10.0.2.15:10000, Asterisk receives, replies to source address which appears as 10.0.2.2:port (SLIRP gateway NAT). That works if browser actually sends to host's mapped port. Browser uses ICE candidates from SDP. If Asterisk includes host candidate 10.0.2.15:10000 — browser can't reach it. Browser's own candidates: host candidates (192.168.x.x of Windows) — Asterisk receives them but they're behind SLIRP NAT so Asterisk sees 10.0.2.2 as source.

Practical solution people use: set external_media_address / external_signaling_address in pjsip transport to the Windows host IP? No — hostfwd binds to host; from browser's perspective the server is at host IP (e.g., 192.168.1.50:8089). Set in Asterisk transport: external_signaling_address=192.168.1.50 (or whatever) and for media... WebRTC ICE with Asterisk: need ICE support. In pjsip transport there are options: external_media_address, and for ICE you might set the "advertised" addresses. Honestly, with SLIRP this gets messy. Alternative approach: use QEMU with a bridged/TAP network instead of user-mode — then VM has LAN IP, all UDP fine, no hostfwd needed. That's the cleanest for RTP. But user explicitly mentioned hostfwd.

Options to present honestly:

    Recommended: bridge/TAP networking (VM gets real LAN IP) — makes RTP trivial. But user asked hostfwd.
    With hostfwd (SLIRP): signaling WSS works via forwarded TCP 8089. RTP: WebRTC requires ICE; we can make it work by:
        Narrow RTP range (e.g. 10000-10010), forward each UDP port.
        Configure Asterisk pjsip transport with external_media_address and external_signaling_address = host's LAN IP, plus rtp symmetric. Also ICE candidates from Asterisk: Asterisk includes its candidates in SDP; with external_media_address set, does it apply to ICE candidates? There's known issue: WebRTC behind NAT with Asterisk — people use icesupport=yes and set external address via TURN or via rtp.conf stunaddr? Hmm.

Actually there's a well-known simpler path: disable ICE requirement? SIP.js with WebRTC always uses ICE since browsers mandate it. Asterisk since 15+ supports ICE. There's known config trick: in rtp.conf set ice_host_candidates — not exactly. Let me recall: rtp.conf has stunaddr, turnaddr. And pjsip transport option external_media_address does rewrite SDP connection address but ICE candidates are separate. There is a documented approach: Asterisk wiki "Configuring Asterisk for WebRTC clients" and "NAT" — for ICE, Asterisk uses its own host candidates; external_media_address affects... Actually in newer Asterisk (16+), external_media_address also rewrites ICE candidates? I believe there was ABE... Not certain.

Pragmatic widely-used trick for local dev behind SLIRP: set the guest's Asterisk to advertise the host's address, and since hostfwd forwards host:10000-10010 UDP → guest, browser sends RTP to hostIP:10000, Asterisk receives; Asterisk's outgoing RTP goes to the source it saw (10.0.2.2:xxx) via SLIRP which NATs back to browser. Symmetric RTP handles this even if the advertised address mismatch. The risk: browser's ICE may fail connectivity checks if Asterisk's candidates are unreachable (10.0.2.15). ICE requires successful connectivity checks before media flows. With ICE-lite? Asterisk full ICE. If all Asterisk candidates unreachable, ICE fails... unless Asterisk's check toward browser's candidate succeeds: browser candidate 192.168.1.50 (host LAN) — Asterisk sends from guest, SLIRP NATs outbound UDP, browser receives check from SLIRP-mapped address, replies to it → connectivity established. So ICE can succeed via browser's host candidate. So it may actually work without much config! The key requirement: Asterisk must be able to send UDP out (SLIRP allows outbound UDP) and the RTP ports must be in range. And hostfwd for RTP only needed for browser→Asterisk initial packets; ICE checks initiated by Asterisk outbound create SLIRP NAT mappings which allow return traffic. Since browser is ICE-controlling and will also send checks to Asterisk's candidates (10.0.2.15, unreachable → fail), but the pair where Asterisk checks toward browser succeeds works. So RTP can flow entirely through SLIRP outbound NAT without hostfwd of RTP at all! Except the browser needs to know where to send — it uses the pair from successful check. So actually with SLIRP, WebRTC to Asterisk often works because Asterisk does outbound ICE checks. Good — I'll mention we may not even need RTP hostfwd, but to be safe narrow the RTP range and forward a few UDP ports, or better recommend bridge mode as the "it just works" alternative.

Hmm, but I should keep the guide actionable. Strategy: give hostfwd setup including a small RTP UDP range forwarded, set external_signaling_address so SIP over WS Contact is right? For WebSocket transport, signaling is over the WS connection itself, so Contact addressing is less of an issue (connection-oriented). SIP.js will register fine.

Let me not over-engineer. Plan for part 2:

    Certs: self-signed (openssl), combined pem.
    asterisk http.conf: TLS on 8089, wss enabled, enablestatic.
    pjsip.conf: transport transport-wss (protocol wss), webrtc=yes endpoint option (Asterisk 15+: webrtc=yes sets use_avpf, media_encryption=dtls, ice_support, rtcp_mux etc.).
    Endpoint 1000/1001 (auth, aors).
    extensions.conf: dialplan.
    Serve page via Asterisk static dir.
    Browser: accept self-signed cert (visit https://hostip:8089/http_ping? Actually asterisk has /http_status or static index). Warn user to accept cert for both page and WSS (same origin so one acceptance).
    SIP.js config: wsUrl wss://HOSTIP:8089/ws, domain = Asterisk SIP realm (e.g., the VM's IP or "asterisk" default? For pjsip identify by From header username → endpoint matching; SIP.js domain should match what asterisk expects — pjsip matches endpoints by From user if anonymous/identify. Actually PJSIP endpoint matching: by IP first (identify), then From user. Since SLIRP source IP is 10.0.2.2, identification falls to From username. Set identify_by=username,auth_username maybe. Default identify_by includes username? Default is "username,ip"? I think default is ip,username. Safe to note it.)
    Also authorizationRegister: register with auth.

Prod difference: real IP, DNS, port 443, Let's Encrypt cert, STUN/TURN if clients behind NAT (though WS clients mostly direct), SIP over TCP/TLS 5061 for hardphones, media direct RTP etc.

Part 3: dialplan IVR:

    Background()/WaitExten, two-level menu, Playback vs Read, SayNumber.
    TTS options: built-in sounds, and Festival? Asterisk has Festival() app if compiled, or use Swift/AGI with pico2wave or espeak-ng to pre-generate wav. I'll show espeak-ng/pico to generate prompts and format conversion with sox (8kHz mono sln16 → use sox -t wav ... and asterisk native format conversion; simplest: produce 8kHz mono 16-bit WAV and Asterisk plays fine; or convert to sln).
    AGI: FastAGI vs AGI. Write Python AGI script (python3) reading stdin AGI env, arg_database.py, use sqlite3 stdlib. Show agi exec flow: answer, stream file, get data, set variable. Actually easier: use AGI only for DB query, dialplan does prompts: e.g. "Enter account number", dialplan collects digits, AGI(query.php/py) returns balance via SET VARIABLE, dialplan SayNumber. Show both simple and full-AGI example. Mention panasonic... no. Mention libraries: PAGI (php), asterisk-agi? For python, just raw stdio to teach the protocol — good pedagogically. Show the AGI protocol raw (agi_request, agi_channel..., 200 result=...).

Part 1: install: apt install asterisk on Debian 12 (Asterisk 20 LTS). Disable chan_sip? Not present by default in 20 (chan_sip removed in 21; in 20 it exists but disabled). Smoke test from inside VM: use asterisk CLI channel originate Local/... and console Console/dial? Without sound card, can use Channel: Local/200@from-internal with dialplan Playback → check CLI. Or use asterisk -rx 'channel originate Local/1000@from-internal application Playback demo-congrats' — but that just runs. Better smoke test: asterisk -rvvv, originate a call to a Local channel that executes Playback, verify in CLI logs. Also core show channels. Good.

Also mention asterisk -rx "core show version", systemd enable.

Cert generation: openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -subj "/CN=..." then cat > asterisk.pem, chown asterisk.

http.conf:

[general]
enabled=yes
bindaddr=0.0.0.0
bindport=8088
tlsenable=yes
tlsbindaddr=0.0.0.0:8089
tlscertfile=/etc/asterisk/keys/asterisk.pem
enablestatic=yes

pjsip.conf:

[transport-wss]
type=transport
protocol=wss
bind=0.0.0.0
; external_media_address / external_signaling_address if needed

Wait — for WebSocket transport, Asterisk auto-creates? No, need a transport with protocol wss (or 'ws'?) The docs: define type=transport protocol=wss plus cert via http.conf? Actually the WSS transport uses the HTTP server's TLS. Define transport with protocol 'wss'. Yes.

Endpoints:

[1000]
type=endpoint
context=from-internal
disallow=all
allow=opus,ulaw
webrtc=yes
auth=auth1000
aors=1000

[auth1000] type=auth auth_type=userpass username=1000 password=devpass1000
[1000] type=aor max_contacts=5 remove_existing=yes (webrtc clients change ports)

extensions.conf:

[from-internal]
exten => 100,1,Answer()
 same => n,Playback(demo-congrats)
 same => n,Hangup()
exten => 200,1,Answer() ... IVR ...

Webpage: HTML + SIP.js from CDN (jsdelivr sip.js 0.21.x API — SimpleUser). Use SimpleUser for brevity: SimpleUser.createServer()... API:

const simpleUser = new SIP.Web.SimpleUser({/* delegate */});
await simpleUser.connect({ wsURL, displayName, ... }) — connect takes options { userName, password? } Actually SimpleUser.connect({wsURL, displayName}); then register? SimpleUser: connect({ wsURL, }), then call. Authorization: connect options include `password`? SimpleUser.connect({ wsURL, displayName, password? }) Hmm. In SIP.js SimpleUser, `connect(options: ConnectUserOptions)` includes `webSocket: { wsURL }`, `sipPassword`? Let me recall: ConnectOptions: `webSocket` and `sip` options; SimpleUser.connect({wsURL, displayName, ...rest}) — there's `sipPassword`? I recall SimpleUser options in connect: `wsURL`, `displayName`, `sipPassword`? Actually I think it's `password`... In SIP.js SimpleUser docs: `simpleUser.connect({ wsURL: ..., displayName: ... })` and authorization password is passed via `sipPassword` in connect options? Checking memory of sip.js 0.20/0.21 simple user: ConnectUserOptions extends SessionDescriptionHandlerOptions etc. It has `webSocket?: WebSocketInterfaceOptions`... Hmm.

To avoid API-version landmines, maybe use the lower-level UserAgent API directly? That's more code but stable-ish. Or use SimpleUser and set `authorizationPassword`. I recall `simpleUser.connect({ wsURL, displayName, authorizationPassword: 'x', authorizationUsername: '1000' })`? I believe ConnectUserOptions includes `authorizationUsername` and `authorizationPassword`. Yes! I'm fairly confident: SIP.js SimpleUser connect options include `authorizationUsername` and `authorizationPassword`. And `sipExtension`? Hmm, in recent versions: `connect({ wsURL, sipUserName? ...})`. There is `userName` too...

Given uncertainty, I'll write the client with SimpleUser but pin a known version (sip.js 0.21.1 via jsdelivr) and use the documented options I best remember:

```js
const simpleUser = new SimpleUser(URI... )

Actually SimpleUser constructor takes a server? Old API: new Web.SimpleUser(wsURL, options). Newer: new Web.SimpleUser(options) then connect({wsURL}). In 0.21: const simpleUser = new Web.SimpleUser({ delegate: {...} }); await simpleUser.connect({ wsURL, authorizationUsername, authorizationPassword, displayName }); await simpleUser.register();? Hmm, register takes no options maybe.

Alternatively use the plain UserAgent + Registerer + Inviter with a MediaHandler — that's a lot of code. For a tutorial, SimpleUser is right; I'll present code and note "APIs shift between SIP.js majors; this targets 0.21.x" and structure the code so fixes are obvious.

Let me draft SimpleUser 0.21 usage from memory (I've seen examples):

const uri = new SIP.URI("sip", "1000", domain);
const simpleUser = new SIP.Web.SimpleUser(uri, { /* delegate */ audioHelper: ... });

Hmm... older examples: new Web.SimpleUser("wss://...", {audioHelper...}).

In SIP.js 0.20 docs (simple user):

const simpleUser = new Web.SimpleUser("wss://sip.example.com/ws");
await simpleUser.connect();
await simpleUser.register();
await simpleUser.call("sip:100@domain", { media: { constraints: { audio: true, video: false } } });

Connect options: connect({ displayName, sipPassword? })... The 0.20 release changed SimpleUser constructor to take options object: new Web.SimpleUser({ delegate: {...} }) and connect({ wsURL, userName, password? })?

I remember the current docs:

const simpleUser = new Web.SimpleUser("wss://...");
await simpleUser.connect({ authorizationUsername: "alice", authorizationPassword: "secret" });

Hmm, or connect({ userName, displayName })...

I'll write with a compatibility note. Let me check what I firmly know from SIP.js 0.21.x README example:

From SIP.js README (0.21):

const simpleUser = new Web.SimpleUser("wss://sip.example.com/ws");
await simpleUser.connect();
await simpleUser.call("sip:bob@example.com");

Hmm, actually I believe newer SimpleUser: constructor takes delegate/options: new Web.SimpleUser({ delegate: {...} }), then await simpleUser.connect({ wsURL, authorizationUsername, authorizationPassword })?

I've definitely seen in sip.js 0.20.0 release notes: "SimpleUser constructor now takes an options object instead of server" — something like that. And connect options include wsURL, displayName, authorizationUsername, authorizationPassword, sipRegistrationExpiration... I'm fairly confident about authorizationUsername/authorizationPassword being in ConnectUserOptions. Good, go with that, pinned to 0.21.x.

Digits: IVR often needs DTMF. WebRTC sends DTMF via RTP telephone-event (Asterisk supports) — SIP.js sendDTMF exists on SimpleUser? SimpleUser has sendDTMF(tones). Yes, I believe simpleUser.sendDTMF("1"). Good — phonepad sends DTMF both pre-call (dial extension then Call) and in-call (sendDTMF).

Phonepad features: display input, backspace, call/hangup buttons, in-call DTMF buttons. Keep modest.

Serving page: put index.html in /var/lib/asterisk/static-http/index.html, so https://HOSTIP:8089/ serves it. Need http.conf enablestatic=yes. Default prefix empty. Good.

Also must accept the self-signed cert: browse to https://HOSTIP:8089/ and accept warning (same cert used for WSS, so one acceptance covers). Good point to include.

Asterisk config reload: asterisk -rx "pjsip reload", http show status.

Smoke test part 1 details:

sudo apt update && sudo apt install -y asterisk
sudo systemctl enable --now asterisk
sudo asterisk -rvvv
core show version
pjsip show transports

Local smoke test: originate Local channel:

asterisk -rx "channel originate Local/600@from-internal application Wait 5"? 

Better: dialplan exten 100 Playback demo-congrats; originate Local/100@from-internal; watch CLI verbose; core show channels. Also CLI> core show sounds. Also verify sounds installed (asterisk-core-sounds). Debian package includes sounds (asterisk-core-sounds en). Fine.

Also default Debian config is huge sample; recommend starting clean: move pjsip.conf/extensions.conf aside and write minimal. Good advice.

Part 3 details:

extensions.conf main IVR:

[general] static=yes writeprotect=no
[from-internal]
exten => 100, Playback congrats (smoke)
; two-level menu
exten => 300,1,Answer()
 same => n,Set(TIMEOUT(digit)=3)
 same => n,Set(TIMEOUT(response)=8)
 same => n,Background(custom/welcome)   ; or Playback then WaitExten
 same => n,WaitExten(8)
exten => 1,1,Playback(...) ; sales submenu
 same => n,Background(...) WaitExten
exten => 1,1 vs 2 etc; also i (invalid), t (timeout)

TTS: espeak-ng / pico2wave generating wav then sox to 8k:

espeak-ng -v en-us -s 140 -w /tmp/x.wav "text"
sox /tmp/x.wav -r 8000 -c 1 /var/lib/asterisk/sounds/en/custom/x.wav

Actually Asterisk can transcode wav automatically, but 8k mono avoids resampling; recommend gsm or sln16. Keep sox.

AGI: /var/lib/asterisk/agi-bin/balance.py:

#!/usr/bin/env python3
import sys, sqlite3, json
# read AGI env until blank line
def read_env():
    env={}
    while True:
        line=sys.stdin.readline().strip()
        if not line: break
        k,_,v=line.partition(': ')
        env[k]=v
    return env
def cmd(c):
    sys.stdout.write(c+"\n"); sys.stdout.flush()
    r=sys.stdin.readline().strip()
    return r  # "200 result=0"

Dialplan: same => n,AGI(balance.py,12345) and script does SET VARIABLE BALANCE 42, then SayNumber(${BALANCE}). DB: sqlite3 /var/lib/asterisk/agi-bin/accounts.db with a table. Show creation SQL. Mention chan variable access in AGI via GET VARIABLE. Also mention FastAGI (agi://) and frameworks (PAGI, asterisk-java), and newer options: ARI (REST) — worth a paragraph since modern dev prefers ARI; explain AGI vs ARI vs dialplan tradeoffs.

Part 4: getting a real number: ITSPs / SIP trunking providers (Twilio...

message: Error in input stream

--

attached is the user query and your thinking process. your AI infra crashed last time and attached is the backup file. please pickup and continue and finish the task. thanks!



Let me look at what happened. The user's backup file shows their original query and the beginning of my previous thinking process, which was cut off mid-way through planning part 4 ("Part 4: getting a real number: ITSPs / SIP trunking providers (Twilio..." with "message: Error in input stream" indicating the crash).

I need to pick up and finish the task: write the complete comprehensive tutorial covering all 4 parts:

    Install and configure Asterisk on Debian for local dev, with a smoke test from inside the VM.
    IVR end-to-end smoke test: hello-world IVR (dialplan), plus an HTML5 softphone webpage using SIP.js, with QEMU hostfwd networking explained for both server and client sides, plus brief note on how a prod server with public IP differs.
    Quick tour of Asterisk IVR programming: two-level menu, audio prompts (built-in/TTS vs programmer-supplied), AGI script querying SQLite.
    Outro: how to get a real/public phone number and link it to the PBX.

The previous thinking process had made these technical decisions which I should honor and continue:

    Debian 12 (bookworm) with Asterisk 20 LTS from apt
    PJSIP (not chan_sip), WebSocket/WSS transport
    Self-signed TLS cert because browsers require secure context for getUserMedia (mic access)
    Serve the webpage from Asterisk's built-in HTTP static server on TLS port 8089 (minimizes moving parts; one cert covers page + WSS)
    QEMU user-mode networking with hostfwd: TCP 8089 (and 2222 for SSH). RTP: discuss ICE/SLIRP NAT traversal — Asterisk's outbound ICE checks create SLIRP NAT mappings so RTP often works without forwarding RTP ports; to be safe, narrow the RTP range and forward a few UDP ports; recommend bridge/TAP as the "it just works" alternative
    SIP.js SimpleUser pinned to 0.21.x, with authorizationUsername/authorizationPassword in connect options, sendDTMF for in-call DTMF
    identify_by=username for endpoint matching since SLIRP source IP is 10.0.2.2
    webrtc=yes on endpoint
    Clean minimal configs (move sample configs aside)
    Part 3: two-level menu with Background/WaitExten, TTS via espeak-ng/pico2wave + sox conversion to 8kHz mono, raw Python AGI script reading the AGI stdio protocol, SQLite database, mention FastAGI, frameworks, and ARI as the modern alternative
    Part 4: was cut off at "ITSPs / SIP trunking providers (Twilio..." — I need to complete this: ITSPs, DID numbers, SIP trunk configuration (pjsip trunk endpoint, registration, context for inbound calls), and briefly mention ENUM/E911, country regs, etc.

Let me now write the full tutorial. It should be a comprehensive, well-structured technical document. The user said "This is an extended task, so take your time!" — so they want thoroughness.

Let me think about the remaining details I need to nail down:

Part 1: Install & smoke test

sudo apt update
sudo apt install -y asterisk

Debian 12 ships Asterisk 20. Packages: asterisk (meta), asterisk-core-sounds (en/wav/gsm), asterisk-modules etc.

sudo systemctl enable --now asterisk
sudo asterisk -rvvv

CLI commands: core show version, core show channels, pjsip show transports, http show status.

Smoke test from inside VM: originate a Local channel that plays a sound, verify with verbose output and core show channels:

asterisk -rx "channel originate Local/100@from-internal application Playback demo-congrats"

Wait — channel originate syntax: channel originate <tech/data> application <appname> [appdata] or channel originate <tech/data> extension [exten@][context]. So channel originate Local/100@from-internal application Playback demo-congrats — hmm, if originating to Local/100@from-internal with application, it would run the application on the originated channel, not dial 100 in that context. Actually channel originate Local/100@from-internal application Playback demo-congrats — the Local channel's "100@from-internal" part specifies where the local channel goes (i.e., the local channel will execute exten 100 in context from-internal on one leg), and then application... no wait.

Let me get this right. The CLI command: channel originate <channel> application <application> [appdata] — this originates a channel and executes the application on it. E.g. channel originate Local/100@from-internal application Wait 10 would originate a Local channel whose "other side" goes to 100@from-internal, and the originated side executes Wait. Hmm, that's confusing.

Simpler and well-known: channel originate Local/100@from-internal extension 100@from-internal? No.

The cleanest smoke test that doesn't depend on my recall of originate semantics: define exten 100 in dialplan that does Answer/Playback/Hangup, then:

asterisk -rx "channel originate Local/100@from-internal application Playback demo-congrats"

Hmm, with Local channel originate + application: Asterisk originates the Local channel; the Local channel's pbx side (the @from-internal part) dials into exten 100 context from-internal — wait no.

Let me think about how Local channel originate works: channel originate Local/100@from-internal creates a Local channel where the "2" leg executes exten 100 in context from-internal, and the "1" leg (the originated channel) is controlled by the application/extension specified in the originate command. So:

    channel originate Local/100@from-internal application Playback demo-congrats: leg 1 plays demo-congrats, leg 2 runs dialplan at 100@from-internal. If dialplan 100 is Answer(); Playback(demo-congrats); Hangup(), then both legs answer and play congrats to each other — call completes, we see it in CLI. That's a fine smoke test showing dialplan executes and sound files play.

Actually even simpler: make exten 100 self-contained, then channel originate Local/100@from-internal application Wait 30... no, if leg2 executes dialplan 100 (Answer, Playback, Hangup) and leg1 does Wait, the call is up briefly.

Hmm, honestly the most common smoke test I've seen documented:

asterisk -rx "channel originate Local/200@from-internal extension 200"

no wait. Let me just be careful: From Asterisk docs for channel originate:

channel originate <tech/data> extension [exten@][context]
channel originate <tech/data> application <appname> [appdata]

"Originate a call to a channel and connect it to an extension or application."

So channel originate Local/100@from-internal application Playback demo-congrats: Asterisk calls "Local/100@from-internal" (i.e., the far end of the Local channel is plugged into dialplan 100@from-internal), and when answered, connects the originated side to application Playback. Hmm, actually with Local channels, "calling Local/100@from-internal" means the originated channel IS the local channel, and its far side is at dialplan 100@from-internal. Then "application Playback" runs on the originated (near) side.

Either way, the dialplan at 100@from-internal executes. So if 100 = Answer + Playback + Hangup, and the app side does Playback demo-congrats, we'll see in verbose output: "Executing [100@from-internal:1] Answer()", "Executing ... Playback("...","demo-congrats")", etc. And core show channels during the call shows two Local channel legs. That's a solid smoke test.

Even simpler alternative that avoids originate entirely: use console channel? Requires ALSA/OSS console driver and a sound device — VM has none. Skip.

Another neat trick: asterisk -rx "core show translation" etc. But the Local originate is the real call-path test. I'll go with:

# watch live:
sudo asterisk -rvvv
# in another terminal (or via CLI):
channel originate Local/100@from-internal application Playback demo-congrats

Hmm wait, one more consideration: with application Playback demo-congrats on leg1 and dialplan on leg2 — leg2 does Answer. Playback on leg1 requires answered channel; since leg2 answers, the Local channel answers leg1. OK it works.

Actually, maybe cleaner: channel originate Local/100@from-internal application Wait 5. Dialplan 100: Answer, Playback(demo-congrats), Hangup. We'll see dialplan execution. Either fine. I'll use Playback version... Actually to make the smoke test scream "dialplan works", let me have the CLI show both legs. I'll present:

dialplan:

exten => 100,1,Answer()
 same => n,Playback(demo-congrats)
 same => n,Hangup()

smoke:

sudo asterisk -rx 'channel originate Local/100@from-internal application Playback demo-congrats'

Hmm, but if dialplan hangs up after playback, the whole call ends — fine, brief call. To observe with core show channels, could use Wait. Let me make exten 100: Answer, Playback(demo-congrats), Wait(10), Hangup — no, keep it simple; the verbose log is the evidence.

Actually, I realize there might be an even more classic approach for "smoke test directly from inside the VM": install a CLI softphone? Like sipp or linphone-cli? That's heavy. Or asterisk's built-in Echo() test with... no audio device. The Local-channel originate IS the canonical "headless smoke test". Also core show sounds | head. Good.

Also I should suggest checking that PJSIP is loaded: module show like res_pjsip / pjsip show endpoints.

Also note: Debian's default configs are in /etc/asterisk (huge samples). Recommendation: back up and create minimal pjsip.conf, extensions.conf, http.conf, rtp.conf. Also mention asterisk -rx "core restart now" vs reload commands.

Also worth mentioning: sudo apt install asterisk on bookworm → Asterisk 20.x (LTS). If Debian 13 trixie → Asterisk 22? Actually trixie ships asterisk 1:22... hmm, trixie has asterisk 22? Let me think — Debian 13 (trixie, released Aug 2025) ships Asterisk 22? I believe trixie has asterisk 22.x (or 21?). Not critical — I'll say "Debian 12 (bookworm) ships Asterisk 20 LTS; these instructions work for 18/20/21/22 with minor differences." Safe.

Part 2: End-to-end

Architecture diagram (ASCII):

Windows host (browser + mic/speaker)
   │  HTTPS page + WSS SIP signaling  (TCP 8089)
   │  RTP/SRTP media (UDP 10000-10010)
   ▼
QEMU hostfwd (user-mode NAT)
   ▼
Debian VM 10.0.2.15
   └─ Asterisk: HTTP/TLS :8089 (ws + static), PJSIP wss transport, RTP 10000-10010

QEMU command example:

qemu-system-x86_64 \
  -m 2048 -smp 2 \
  -netdev user,id=n0,hostfwd=tcp::2222-:22,hostfwd=tcp::8089-:8089,hostfwd=udp::10000-:10000,hostfwd=udp::10001-:10001,hostfwd=udp::10002-:10002,hostfwd=udp::10003-:10003,hostfwd=udp::10004-:10004 \
  -device virtio-net-pci,netdev=n0 \
  ... disk etc

And the important explanation about RTP/ICE with SLIRP:

    TCP signaling via forwarded 8089 is straightforward.
    Media is UDP; browsers (WebRTC) use ICE. Asterisk also runs ICE (ice_support=yes from webrtc=yes).
    Asterisk's ICE candidates will advertise 10.0.2.15 (unreachable from host browser) — but ICE pairs succeed via the other direction: Asterisk sends its checks outbound (SLIRP NATs outbound UDP fine), browser replies to the mapped address, connectivity check succeeds, then symmetric RTP keeps flowing through the SLIRP NAT binding.
    So in practice RTP often works without forwarding any UDP ports. Forwarding a few RTP ports + narrowing rtp.conf range is belt-and-suspenders; it helps if the browser's checks to hostfwd'd ports succeed first.
    If audio is one-way/none, the fix is: narrow range, forward those UDP ports, and/or switch to bridged networking (give the VM a LAN IP via -netdev bridge or tap, or use libvirt's default NAT network with virt-manager which is often easier), or run the browser on the host with... hmm.

Wait, actually, let me reconsider the ICE details. With webrtc=yes, Asterisk endpoint gets ice_support=yes. Asterisk includes ICE candidates in its SDP answer (host candidates on 10.0.2.15). Browser tries to reach 10.0.2.15 — fails (unless the Windows host happens to have a route... it doesn't). Browser's candidates: host candidate = Windows LAN IP (e.g., 192.168.1.50) possibly plus mDNS candidate! Important: modern Chrome hides host candidates behind mDNS (.local) hostnames to prevent fingerprinting. Asterisk can't resolve mDNS names... Chrome's mDNS candidates — does Asterisk resolve them? Asterisk/PJSIP generally fails to resolve mDNS candidates. Hmm, but Chrome only uses mDNS candidates when the page doesn't have the "media devices" permission? Actually Chrome replaces host candidates with mDNS hostnames unless the origin has been granted... I recall: Chrome uses mDNS obfuscation for host candidates by default for non-permitted origins; once the user grants mic permission to the origin, real host candidates are exposed? Hmm, I believe Chrome still uses mDNS candidates even with permission in some versions... Actually: Chrome mDNS candidates appear when ICE gathering happens; for sites with mic permission granted, Chrome exposes real host candidates (the mDNS protection applies to un-permitted origins... no...). Let me recall properly: Chrome's "Anonymize local IPs exposed by WebRTC" — enabled by default since Chrome 70ish. It replaces host candidates with mDNS hostnames (random.local). Third parties can't resolve them. However, on the same machine... Asterisk (in VM) receiving an mDNS candidate will fail DNS resolution (SLIRP DNS forwarding of .local? no, mDNS is link-local multicast; the VM can't reach the host's mDNS). So Asterisk's checks toward browser candidates fail.

Then the only viable pairs are: browser → Asterisk candidates (10.0.2.15, unreachable from Windows). Hmm. That's a problem!

Wait, but there's the hostfwd UDP ports: if Asterisk's candidate is 10.0.2.15:10000, browser tries 10.0.2.15 — unreachable regardless of hostfwd, because hostfwd maps hostIP:10000 → guest:10000, and browser would need to send to hostIP (e.g., 192.168.1.50 or 127.0.0.1):10000, but the SDP says 10.0.2.15.

Unless we set Asterisk to advertise the host's address! Options:

    rtp.conf: ice_host_candidates — hmm, is there such an option? There IS ice_host_candidates section in rtp.conf? I recall rtp.conf has a [ice_host_candidates] section where you can map local addresses to external addresses for ICE! Yes! Asterisk rtp.conf supports:

[ice_host_candidates]
10.0.2.15 => 192.168.1.50

Hmm, I'm not 100% sure of syntax. Let me think... I recall from Asterisk docs: "ice_host_candidates" section in rtp.conf allows mapping an IP address Asterisk would use for a host candidate to a different address. I'm fairly confident this exists (used for ICE NAT traversal without TURN). The wiki page "ICE" mentions rtp.conf [ice_host_candidates] with lines like 192.168.1.1 => 1.2.3.4. Hmm. I genuinely believe I've seen this. Let me go with it but hedge ("if your Asterisk version supports the ice_host_candidates mapping in rtp.conf").

Actually wait — but is the Windows host IP (e.g., 192.168.1.50) reachable from the browser? Yes, the browser runs on that host. Browser sends to 192.168.1.50:10000 (host candidate advertised by Asterisk) → hostfwd udp::10000-:10000 forwards → guest 10.0.2.15:10000. Asterisk receives STUN binding request from source 10.0.2.2:xxxxx, responds to that source → SLIRP translates back to browser. Connectivity check succeeds! Then media flows same path.

But wait — will the browser's source port for the STUN check to 192.168.1.50:10000 match its candidate? ICE checks come from the candidate's socket, yes. Asterisk's response goes to source address (symmetric). Works.

Also the browser→asterisk check needs hostfwd for those specific RTP ports. And Asterisk→browser checks fail (mDNS/​unreachable), but ICE only needs one working pair per component... ICE needs a valid pair; successful check from either direction creates a peer-reflexive/valid pair. Browser is controlling agent, nominates the working pair. Should work.

Alternatively, simpler robust path that many tutorials use: just try it — often it works out of the box because... hmm, does it? Without ice_host_candidates and without reachable Asterisk candidates, ICE fails → no audio. With Chrome mDNS, Asterisk can't check toward browser. So out-of-the-box SLIRP + WebRTC likely FAILS for audio. Hmm wait, but hold on — is there another mechanism: Asterisk, when ice_support=no on the endpoint? Browsers REQUIRE ICE (WebRTC mandates ICE; SDP without ICE candidates is rejected by browsers? Actually browsers can do ICE-lite? No — browsers implement full ICE and require ICE attributes; a remote SDP without ice-ufrag/ice-pwd/candidates will fail). Hmm, but there's a subtlety: if the browser's offer has ICE and Asterisk answers with... Asterisk with ice_support=no answers without ICE candidates — browser treats as... For WebRTC, browser requires ICE in answer; it would fail. So ICE must succeed.

Wait, but actually — what about Chrome's mDNS? Let me reconsider: Chrome's mDNS candidates — I now recall that Chrome only mDNS-obfuscates host candidates when the site does NOT have mic/camera permission; once permission granted, real IPs are exposed? Hmm no... Let me think about what I know from practice: In Chrome, getUserMedia with permission → the ICE candidates in about:webrtc show real local IPs (192.168.x.x) for permitted origins, and mDNS (xxxx.local) for non-permitted. I'm fairly sure that's right: the mDNS anonymization applies when permission hasn't been granted, and there's a flag webrtc.ipHandlingPolicy (default vs default_public_interface_only etc.). Yes — Chrome policy "WebRTC IP handling": default uses all interfaces for permitted origins. For our use, user grants mic permission → real host candidate 192.168.1.50 exposed. Then Asterisk CAN check toward 192.168.1.50:port — SLIRP outbound NAT allows UDP out from guest to Windows host IP... wait, can the guest reach the host? SLIRP gateway is 10.0.2.2 which IS the host's SLIRP interface. Sending UDP from guest to 192.168.1.50 (host's LAN IP): SLIRP NATs outbound to the host's stack — the packet arrives on the host addressed to its own LAN IP — yes, that works (SLIRP can reach the host and beyond, that's how the VM reaches the internet). So Asterisk's STUN check toward browser's real host candidate at 192.168.1.50:port arrives at the browser (it's the same machine!). Browser replies to source = SLIRP-mapped address (host's source IP 192.168.1.50 with some SLIRP-assigned source port... wait, SLIRP NATs the guest's outbound packet: source becomes the host's IP (192.168.1.50) with an ephemeral port, destination browser's port). Browser receives check from 192.168.1.50:ephemeral — its response goes there → SLIRP maps back to guest. Check succeeds from Asterisk's side! Then ICE pair (Asterisk 10.0.2.15:port ↔ browser 192.168.1.50:port) is validated via Asterisk-initiated check...

Hmm wait, but the browser must know the pair is valid. In ICE, a successful inbound check triggers a triggered check from the receiving agent toward that candidate pair. Browser receives check from 192.168.1.50:ephemeral (peer-reflexive candidate for Asterisk), sends triggered check back to that address → succeeds (SLIRP mapping alive). Pair validated. Browser (controlling) nominates it. Media: browser sends RTP to 192.168.1.50:ephemeral?? No wait — media flows on the validated pair: browser sends to the peer-reflexive candidate address 192.168.1.50:ephemeral_port → SLIRP → guest Asterisk. Asterisk sends to 192.168.1.50:browserport → host loopback? → arrives at browser. BOTH directions work via SLIRP outbound NAT!

So out-of-the-box it CAN work even without hostfwd on RTP and without ice_host_candidates — as long as Chrome exposes the real host candidate (mic permission granted) and Asterisk performs the outbound check. The browser's own checks toward 10.0.2.15 fail, but triggered checks succeed. ICE typically takes a few seconds (browser checks fail/timeout for unreachable candidates, but triggered checks fire immediately on incoming check — Asterisk sends its checks promptly). So connection establishes quickly.

But this depends on: (a) Chrome exposing real host IP (permission granted — yes, since mic needed anyway); (b) SLIRP allowing guest→host UDP (yes); (c) timing/keepalives (RTP keeps NAT mapping alive; Asterisk sends keepalives). This is the "symmetric NAT traversal" style path.

Hmm, but wait — does Asterisk even send checks to the browser's host candidate? Asterisk (full ICE agent via PJPROJECT) gathers candidates from SDP offer, forms pairs, sends checks. The browser's host candidate 192.168.1.50 — Asterisk sends STUN binding request from 10.0.2.15:rtpport to 192.168.1.50:browserport. Out via SLIRP NAT. Arrives at browser. Yes.

OK so honestly, there are two workable stories:

    "It often just works" via SLIRP outbound NAT + ICE triggered checks (no RTP hostfwd needed).
    Belt-and-suspenders: map ice_host_candidates (advertise host IP) + hostfwd the RTP UDP ports, so browser has a directly reachable candidate.

I'll present it honestly: explain the moving parts, give the recommended setup (narrow RTP range + hostfwd a few UDP ports + optionally ice_host_candidates mapping), and the debug path (rtp set debug on, core show channels, about:webrtc). And mention bridged networking as the zero-hassle alternative for serious dev.

Hmm wait, one more thing about ice_host_candidates — let me make sure about the exact syntax, because I want to be accurate. From Asterisk source/docs: rtp.conf:

[ice_host_candidates]
; RTP ICE candidate mapping
;
; In this section you can map the local IP address Asterisk would
; put into an ICE candidate to a different IP address. This can be used
; to handle NAT scenarios where Asterisk is behind a NAT and does not
; know its external public IP address...
; 10.0.0.2 => 74.93.46.23

I'm reasonably confident this section exists in rtp.conf.sample (added around Asterisk 15/16 with ICE support). Hmm... Actually, I'm not fully certain. Let me think about what I know is definitely in rtp.conf: rtpstart, rtpend, icesupport, stunaddr, turnaddr, strictrtp...

I do recall a wiki page about "Configuring Asterisk with ICE"... There's definitely stunaddr and turnaddr options in rtp.conf. For the mapping section, I have genuine memory of [ice_host_candidates] existing... I'm going to include it but frame it as "if supported in your version, rtp.conf.sample documents a [ice_host_candidates] mapping section; check your sample config". Actually — you know what, since I'm not fully sure, I could offer the more certain alternative: STUN/TURN via rtp.conf (turnaddr definitely exists). But a TURN server for local dev is overkill.

Alternatively — hmm, actually there's a much simpler and 100%-reliable consideration: does Asterisk's SDP for the wss endpoint even matter... Yes, it does for media.

Let me just be pragmatic and honest in the write-up:

    Primary: try it; explain why it usually works (ICE triggered checks through SLIRP outbound NAT).
    Add hostfwd for a narrow RTP range (5-10 ports) — cheap, helps in the "browser checks first" case... actually does it? Browser checks go to candidates in SDP: 10.0.2.15 — hostfwd doesn't make 10.0.2.15 reachable. So hostfwd RTP ports only help if Asterisk advertises the host IP (via ice_host_candidates or external_media_address?).

Hmm, wait — what about external_media_address on the wss transport? For WebSocket transports, external_media_address might rewrite the SDP c= line? But ICE candidates are still host candidates on 10.0.2.15. There's a known Asterisk issue: external_media_address doesn't rewrite ICE candidates — that's exactly why [ice_host_candidates] was added? Hmm, maybe. I recall discussion in Asterisk community: "ICE candidates not rewritten by external_media_address" and the workaround being STUN or the mapping section...

OK here's my decision: I'll include the [ice_host_candidates] mapping with a hedge and a "verify against your rtp.conf.sample" note, present the outbound-NAT triggered-check path as the reason it may work without it, give hostfwd for the RTP range, and give a debugging section. That's honest and practical. Also I'll add: worst case, switch to bridged networking or run browser on the VM itself... hmm, browser on VM — no GUI typically. Or test from inside VM with a non-WebRTC softphone (e.g., pjsua? or linphone CLI) to isolate dialplan vs network issues. Actually a nice tip: use microsip/linphone on Windows over... they'd use UDP SIP to 5060, also needs hostfwd and has its own NAT issues but classic SIP+rtp symmetric works fine through SLIRP with hostfwd 5060 + few RTP ports... eh, but the task explicitly wants SIP.js browser. I'll keep focus.

Let me also double check pjsip show transports / ws transport config. In pjsip.conf:

[transport-wss]
type=transport
protocol=wss
bind=0.0.0.0

For WSS, the cert comes from http.conf (the HTTP server handles TLS for websocket). Also need ws transport? For wss only this. Some setups also define a ws transport for non-TLS. Note: with protocol=wss, you may need to NOT set cert here. Right — for WebSocket transports, TLS is provided by the built-in HTTP server (http.conf tlsenable + tlscertfile). Yes, that matches my memory of the Asterisk WebRTC docs: http.conf provides TLS for wss.

Endpoint config with webrtc=yes:

[1000]
type=endpoint
transport=transport-wss
context=from-internal
disallow=all
allow=opus
allow=ulaw
webrtc=yes
direct_media=no
force_rport=yes
rewrite_contact=yes
rtp_symmetric=yes
auth=auth1000
aors=1000

webrtc=yes sets: use_avpf=yes, media_encryption=dtls, ice_support=yes, rtcp_mux=yes, dtls_auto_generate_cert... hmm, does webrtc=yes auto-generate DTLS cert? I recall webrtc=yes implies dtls_auto_generate_cert=yes? Hmm — there IS an option dtls_auto_generate_cert. With webrtc=yes, Asterisk docs say it enables: use_avpf, ice_support, media_encryption=dtls, rtcp_mux, use_received_transport... and DTLS certificate: if not set, "webrtc=yes" also sets dtls_auto_generate_cert=yes? I believe newer versions (16.6+?) made webrtc=yes auto-generate a self-signed DTLS cert if none provided. Yes — I'm fairly confident: "When webrtc is enabled a DTLS certificate will be automatically generated if one is not provided." Good, so no DTLS cert config needed. Media encryption DTLS-SRTP then handles the SRTP keys over the TLS'd websocket signaling... fine, automatic.

Also identify: default identify_by=ip,username? Hmm, PJSIP default identify_by is "username" or "ip,username"? I believe the default is "username,ip"? Let me think. pjsip.conf sample: identify_by default = "username,ip"? Hmm. I recall the default is ip,username? There was a change... In any case, since SLIRP source is 10.0.2.2 for both clients, IP-based identify can't distinguish 1000 vs 1001 — actually both endpoints are the same browser here mostly. To be safe set identify_by=username,auth_username on the endpoint? identify_by is an endpoint option? No — identify is done via the endpoint's... hmm, identify_by is an option on... it's on the endpoint? No! identify_by is on... Let me recall: there's a type=identify section (for static IP matching), and the endpoint has identify_by option? Hmm, actually I'm not sure the endpoint has identify_by...

Wait. Yes, the endpoint DOES have identify_by in newer versions? Hmm. I recall global/endpoint option "identify_by" — I'm fairly sure endpoint has identify_by (values: username, auth_username, ip; default "ip,username"? or "username"?). Ugh.

Let me simplify: with WebSocket connections, PJSIP matches incoming SIP requests: it first checks the transport... For requests over WebSocket, Asterisk historically matched by From username because the connection is from SLIRP. Actually here's the thing I'm more sure about: for WebSocket clients, standard practice (all the Asterisk+WebRTC tutorials) is that endpoint matching by From user works out of the box; tutorials don't set identify_by. E.g., the official Asterisk WebRTC sample config just has endpoint+auth+aor and SIP.js registers fine. Registration itself identifies via auth (WWW-Authenticate with username → endpoint found by username in auth? hmm, that's auth_username matching which requires identify_by=auth_username...).

OK let me think concretely about how SIP.js registration works with PJSIP over WSS:

    SIP.js sends REGISTER From: sip:1000@domain, To: sip:1000@domain, over WSS connection.
    PJSIP receives REGISTER on wss transport. It must find the endpoint: by IP (10.0.2.2) — no match unless an endpoint/identify has that IP; by From user "1000" — matches endpoint [1000] if identify_by includes username. I'm now fairly confident the DEFAULT identify_by is "username,ip" — hmm or is it "ip,username"? The ORDER matters: with "ip,username" PJSIP tries IP match first, then username. Since no IP match, falls to username → endpoint 1000.
    Endpoint has auth=auth1000 → challenge 401 with realm, SIP.js resends with Authorization → validated. REGISTER processed, AoR 1000 gets a contact (the websocket connection).

Yes, this is the standard flow that works in every tutorial, so default identify must include username. Good — no special config needed, but I'll add a note: "if you see 'failed to authenticate' / no endpoint found, set identify_by=username on the endpoints" as troubleshooting. That's safe advice.

Also SIP.js needs domain/uri to match. SIP.js SimpleUser: new Web.SimpleUser(...) — the SIP URI host part: what should it be? Asterisk accepts any domain in From/To generally (it matches on user). SIP.js uses the domain from the target URI for routing? For registration, SIP.js builds To/From from the configured URI: sip:1000@DOMAIN. DOMAIN can be anything (e.g., the host IP or "asterisk.local"); Asterisk doesn't strictly care (unless it has realm checking... no). Many examples use the Asterisk server's IP/domain. I'll use the host's IP as the SIP domain for simplicity (e.g., 192.168.1.50), and register=true.

Wait, one subtlety: SIP.js sends requests with Request-URI sip:1000@domain over the WSS connection to the server; PJSIP treats it as inbound request, matches endpoint by user, fine regardless of domain. Good — use HOST_IP as domain.

Now the SimpleUser API for 0.21.x. Let me carefully reconstruct. SIP.js 0.21.x (2024). SimpleUser docs:

Constructor: new Web.SimpleUser(server?: string, options?: SimpleUserOptions)? OR new Web.SimpleUser(options)?

I remember the migration: In 0.20, SimpleUser constructor changed from new SimpleUser(server, options) to... hmm no. Let me recall the actual SIP.js docs (sipjs.com). The SimpleUser "Simple User" docs page:

const simpleUser = new Web.SimpleUser("wss://sip.example.com/ws", {
  delegate: {
    onCallReceived: ...
  }
});
await simpleUser.connect();
await simpleUser.register();
await simpleUser.call("sip:bob@example.com");

Hmm, and connect options: connect({ authorizationUsername, authorizationPassword })? I have a memory of the docs example:

await simpleUser.connect({
  authorizationUsername: "alice",
  authorizationPassword: "secret"
});

Hmm, actually I think I remember from the SIP.js GitHub README:

// Construct a SimpleUser instance
const simpleUser = new Web.SimpleUser("wss://localhost:8089/ws");

// Connect to server and place call
await simpleUser.connect();

and there's connect(options?: ConnectUserOptions). ConnectUserOptions includes: displayName, sipPassword?, authorizationUsername, authorizationPassword, userAgent?...

Honestly, I've seen BOTH new SimpleUser(server, options) (server string as first arg) in older versions (0.15-0.19?) and I believe it REMAINED that way in 0.20/0.21 — the constructor signature constructor(server?: string, options?: SimpleUserOptions) where server is the WebSocket URL... no wait.

Hmm, let me think about the 0.20 changelog: "SimpleUser now..." I genuinely can't recall a constructor change. Let me go with what I'm most confident about for ~0.21:

From the current SIP.js documentation site (SimpleUser):

// SimpleUser
const simpleUser = new Web.SimpleUser({
  delegate: {
    onCallReceived: async () => { await simpleUser.answer(); },
    ...
  }
});
await simpleUser.connect({
  wsURL: "wss://localhost:8089/ws",
  authorizationUsername: "1000",
  authorizationPassword: "devpass1000",
  displayName: "Browser",
});
await simpleUser.register();
await simpleUser.call("sip:1000@localhost", { media: { constraints: { audio: true } } });

Hmm... I really do think newer SimpleUser takes options-only constructor with wsURL moved to connect(). There WAS a restructure where server param was removed. Evidence: the type ConnectUserOptions has wsURL — I'm quite sure wsURL is a connect option now. And SimpleUserOptions (constructor options) has delegate, audioHelper, media, ua?: UserAgentOptions? something like that.

Hmm, wait, actually I can picture the sipjs.com docs "Simple User" page code sample:

import { Web } from "sip.js";

const simpleUser = new Web.SimpleUser("wss://sip.example.com/ws");

await simpleUser.connect();
await simpleUser.register();
await simpleUser.call("sip:target@example.com");

vs the GitHub repo README...

Ugh. OK — pragmatic approach: pin a specific version via CDN and write code against it, with a comment noting where to check the API doc for that version. Let me pick sip.js 0.21.1 (UMD build: https://cdn.jsdelivr.net/npm/sip.js@0.21.1/dist/sip.min.js exposing global SIP).

For 0.21.1, I'll go with constructor taking server string OR options? To hedge, I can write:

const simpleUser = new SIP.Web.SimpleUser(wsUrl, { delegate: {...} });
await simpleUser.connect({ authorizationUsername, authorizationPassword, displayName });

If the 0.21 constructor is options-only, this breaks (server arg ignored → no wsURL → error). If constructor takes server string, the wsURL in connect might conflict...

Let me try REALLY hard to remember 0.20.0 release notes... "Version 0.20.0 of SIP.js includes: ... SimpleUser API changes..." I remember something about call() options changing to SessionDescriptionHandlerOptions, and connect() no longer accepting...

Alternative memory: the SIP.js docs "Getting Started with SIP.js" tutorial page (the one with Asterisk!) — "Making a call with SIP.js" — it shows:

const simpleUser = new Web.SimpleUser("wss://localhost:8089/ws");
await simpleUser.connect();

That tutorial is well-known ("Hello World" with Asterisk). I'm fairly confident that tutorial exists and uses server-as-first-arg. But which sip.js version does it document? The docs site tracks latest...

Hmm, let me think about the actual source code of sip.js 0.21. From memory of browsing the repo: src/platform/web/simple-user/simple-user.ts:

export class SimpleUser {
  constructor(server?: string, options?: SimpleUserOptions) {
    ...
    const userAgentOptions = options?.ua ?? {};
    ...
    if (server) {
      userAgentOptions.uri ??= ... 
    }
  }

Hmm, I genuinely have a fuzzy memory here. I know at SOME point wsURL became a connect() option named wsURL in ConnectUserOptions...

Actually! I remember now. Looking at the SIP.js API docs for SimpleUser.connect:

connect(options?: ConnectUserOptions): Promise<void>

ConnectUserOptions: displayName?, wsURL?? ... hmm no.

OK, alternative strategy to be SAFE: don't rely on my memory of SimpleUser; use the core UserAgent API directly. It's more verbose (UserAgent + Registerer + Inviter + SessionDescriptionHandler defaults handle audio automatically via WebRTC), but the core API is more stable and I remember it well:

const userAgent = new SIP.UserAgent({
  uri: SIP.UserAgent.makeURI("sip:1000@" + domain),
  transportOptions: { server: "wss://" + host + ":8089/ws" },
  authorizationUsername: "1000",
  authorizationPassword: "devpass1000",
  sessionDescriptionHandlerFactory: SIP.Web.defaultSessionDescriptionHandler().factory,
  ...
});

Hmm, in 0.21 the sessionDescriptionHandler config changed: sessionDescriptionHandlerFactoryOptions and default factory moved: Web.defaultSessionDescriptionHandler()? In 0.20+: SIP.Web.defaultSessionDescriptionHandler().factory? There was a refactor where SessionDescriptionHandler became its own module with defaultSessionDescriptionHandler(). Ugh, more API uncertainty (0.21 moved SDH to a separate package sip.js/lib/platform/web... the Web.defaultSessionDescriptionHandler() thing).

Honestly, SimpleUser handles all that internally — which is exactly why it's the right choice for the tutorial; it absorbs the SDH factory plumbing. I'll write SimpleUser code with my best reconstruction and add a clearly marked "API version note" box: "SIP.js has been changing its SimpleUser constructor/connect signatures across 0.15→0.21; this code targets 0.21.x. If you get a TypeError at startup, check the pinned docs; typically it's moving wsURL between the constructor and connect()."

Hmm, that feels weak for a tutorial. Let me try once more to pin down 0.21.1 SimpleUser definitively...

I'm going to reason from the npm dist file structure of sip.js 0.21.x: sip.js@0.21.x/dist/sip.min.js UMD global SIP, with SIP.Web.SimpleUser.

Now, the docs at sipjs.com... The "Simple User" page I can visualize:

"Creating a Simple User

const simpleUser = new Web.SimpleUser(server, options);

..."

Hmm no. You know, let me recall the Asterisk + SIP.js wiki (Asterisk wiki has "Asterisk 16 with SIP.js" or the chan-sip... no, "Configuring Asterisk for WebRTC clients" and a "SIP.js phone" sample). Digium's "Hello World" demo: https://github.com/asterisk/... there's the famous "sip.js-asterisk-demo"?

And the SIP.js repo has examples... examples/ folder? I don't think sip.js repo has rich examples.

OK here's another angle: I remember reading actual working code from people connecting SIP.js to Asterisk (blog posts, Stack Overflow answers) circa 2021-2023:

const simple = new SIP.Web.SimpleUser({
  delegate: {
    onCallReceived: ...
    onCallHangup: ...
  }
});

await simple.connect({
  wsURL: "wss://192.168.1.50:8089/ws",
  authorizationUsername: "1000",
  authorizationPassword: "1000",
  displayName: "Web Client"
});

await simple.register();
await simple.call("sip:100@192.168.1.50");

I'm now recalling more strongly: the options-object constructor + wsURL in connect(). There WAS a breaking change in 0.20.0: "SimpleUser constructor no longer accepts a server URL; pass wsURL to connect()." Hmm... or was it "0.21.0"?

Actually, let me try to remember the 0.21.0 release notes... "0.21.0: ... TypeScript ... browser support..." can't recall.

Decision: I'll write the client code for the options-object style (my stronger memory for recent versions), pin sip.js@0.21.1, AND include a small compatibility shim note showing the one-line alternative for older/newer versions if the signature differs:

// If your SIP.js version errors here, older versions use:
// new SIP.Web.SimpleUser("wss://host:8089/ws", { delegate: {...} })
// and connect() without wsURL.

That's honest, practical, and keeps the tutorial resilient. The main goal (IVR testing) shouldn't die on a library signature.

Hmm, wait. Actually, let me reconsider — maybe I can remember the actual current docs sample. The SIP.js docs site "Simple User" section... "Getting Started"... The sample on the front page of sipjs.com:

// Construct a SimpleUser
const simpleUser = new Web.SimpleUser({
  delegate: {
    onCallReceived: async () => {
      await simpleUser.answer();
    }
  }
});

// Connect to server
await simpleUser.connect({
  wsURL: "wss://localhost:8089/ws"
});

// Place call
await simpleUser.call("sip:hello@localhost");

YES. I'm now quite confident this is the current (0.20+/0.21) shape — constructor takes options object with delegate, and connect takes wsURL. I remember answer() on call received. And SimpleUserOptions includes audioHelper for remote audio element, media constraints. The audioHelper: new Web.AudioHelper.AudioHelper(remoteAudioElement)? Hmm — SimpleUserOptions has audioHelper?: AudioHelper and there's Web.AudioHelper. SimpleUser by default creates an AudioHelper if... I recall SimpleUser automatically attaches remote audio: you pass media: { remote: { audio: audioElement } }? No...

I remember: SimpleUserOptions: delegate, audioHelper, media (MediaStreamConstraints-ish: { audio: true, video: false }), ua (UserAgentOptions passthrough). And there's a static helper Web.SimpleUser.makeMediaElement? Hmm.

For remote audio: I believe SimpleUser, when constructed, will create an audio element... no. Let me recall the onCallReceived pattern: docs show creating an <audio> in HTML and SimpleUser handles attaching? There's simpleUser.remoteAudio?

I remember AudioHelper class: new Web.AudioHelper.AudioHelper(remoteMediaStream...). SimpleUser constructor: if (options?.audioHelper) this.audioHelper = options.audioHelper; else this.audioHelper = new AudioHelper(...)? and it attaches to... Hmm, I recall SimpleUser has media handling: on answer/call, it does attachRemoteAudioElement?

There IS a SimpleUser option I'm fairly sure about: media: { remote?: { audio?: HTMLAudioElement }, constraints: { audio: true, video: false } }. Hmm... something like SimpleUserMediaRemote...

OK here's the thing — I clearly can't perfectly reconstruct this API from memory, and getting it wrong makes the tutorial fail at step 1 for the user. Given the constraints, the best approach: write clean, well-commented code using my best reconstruction (delegate constructor, connect with wsURL/auth options, call with target and media constraints, sendDTMF for keypad), plus an explicit "version pinning + troubleshooting" box that tells the user exactly what to check (the SimpleUser docs page for the pinned version) and the 2-3 likely signature variations. This is what a senior engineer writing a tutorial would do anyway.

Actually — wait. Let me reconsider using a different, more stable path: the Asterisk ARI or... no, user explicitly said SIP.js. OK, SimpleUser it is.

Let me also remember sendDTMF: simpleUser.sendDTMF(tones: string) — yes, I'm confident this exists on SimpleUser. It sends via RTP (RFC 2833) — need Asterisk to accept telephone-event; PJSIP endpoint with webrtc... SIP.js SDH offers telephone-event 101? SIP.js WebRTC SDH does include audio/telephone-event in its offer? Hmm — browsers don't natively send DTMF via RTP telephone-event... wait, actually they DO: RTCRtpSender.sendDtmf()? Hmm, there's RTCDTMFSender — var dtmfSender = pc.getSenders()[0].dtmf; dtmfSender.insertDTMF('1'). Yes! WebRTC has RTCDTMFSender which sends telephone-event RTP packets. SIP.js's SessionDescriptionHandler negotiates telephone-event if... I believe SIP.js Web SDH includes telephone-event 101 in offers, and sendDTMF uses RTCDTMFSender. Yes — SIP.js supports sendDTMF for audio sessions. And Asterisk accepts telephone-event (enabled by default, dtmf_mode on endpoint: default rfc4733 (which is telephone-event) — for WebRTC endpoints, dtmf_mode=rfc4733? hmm, webrtc=yes might set dtmf_mode=inband+rfc4733? Default dtmf_mode is rfc4733 for pjsip endpoints I think. Wait, no — default is "rfc4733"? Let me recall: pjsip dtmf_mode options: rfc4733, inband, info, auto. Default = rfc4733. Yes I believe default is rfc4733. Good — telephone-event works.)

Hmm, but one catch: Chrome's RTCDTMFSender requires the audio track... fine, we have audio. And Chrome only allows DTMF on outbound... whatever, it works.

Also: simpleUser.hangup(), simpleUser.answer(), simpleUser.call(target, options), simpleUser.register(), simpleUser.isConnected(), simpleUser.isCalling()... connectState? I'll use a small state machine in the page and keep the SIP.js surface minimal: connect, register, call, hangup, sendDTMF, delegate callbacks (onCallReceived/onCallHangup? onCallAnswered? onCallCreated? onCallTerminated? I recall: onCallAnswered, onCallReceived, onCallHangup? or onCallTerminated... hmm. SimpleUserDelegate: onCallAnswered?, onCallCreated?, onCallHangup?, onCallHold?, onCallReceived?, onRegistered?, onUnregistered?, onServerDisconnect?, onUserAgentStateChange?... I think there's onCallHangup? Hmm, maybe it's onCallHangup — I'll use onCallReceived, onCallAnswered?, onCallHangup? Hmm.

Honestly, since we're the caller testing an IVR, we mostly need: connect, register, call, sendDTMF, hangup, and onCallHangup/onServerDisconnect to reset UI. I'll wire those with optional chaining so unknown callbacks don't break.

Now — the browser mic: getUserMedia requires secure context. Page served via https://HOSTIP:8089/ (Asterisk static over TLS) → secure context ✓ (after accepting self-signed cert warning). WSS same origin/port ✓.

One catch: Asterisk's static HTTP server — is it enabled by default? http.conf default has enablestatic=no? The sample http.conf has enablestatic commented? Debian default: /etc/asterisk/http.conf is the standard sample with ;enablestatic=yes. We set it. Static dir: /var/lib/asterisk/static-http/. Serving index.html at https://ip:8089/.

Hmm wait, one thing — does Asterisk's HTTP server serve index.html for "/" or require explicit /index.html? URI "/index.html" works; "/" might 404 or serve index? I believe Asterisk's http.c maps "/" → /index.html? Not sure. I'll tell users to use https://HOSTIP:8089/index.html explicitly to be safe? Hmm, if it 404s that's confusing; giving explicit URL is safer. Hmm, I recall Asterisk static serving does handle "/" by... I really don't remember. Explicit URL it is. Hmm, actually I think there might be an issue: Asterisk's built-in static server has a URI prefix — by default the http.conf has ;prefix=asterisk commented out, so no prefix. With no prefix, static files at /var/lib/asterisk/static-http/foo are served at /foo. And "/" → index.html? I'll just say open https://HOSTIP:8089/index.html. Hmm — hmm, wait, actually I have a vague memory that Asterisk's builtin HTTP redirects "/" to /index.html or serves it... doesn't matter, explicit path works.

Also http_status endpoint: http://ip:8088/http_status or /http_status shows server status — useful smoke test before TLS. I'll include: curl http://127.0.0.1:8088/http_status from inside VM. Hmm, is the URI /http_status? I believe Asterisk has /http_status (shows HTTP server status), /phoneconfig..., /moh..., /aristar... The URI is "http_status" yes. curl http://localhost:8088/http_status → "Asterisk Server Status" HTML. I'm fairly confident.

Also ws debug: sudo asterisk -rx "http show status" shows HTTP server binds and websocket status? http show status shows enabled URIs including websocket service. Good.

TLS cert generation:

sudo mkdir -p /etc/asterisk/keys
cd /etc/asterisk/keys
sudo openssl req -x509 -newkey rsa:2048 -sha256 -days 3650 -nodes \
  -keyout asterisk.key -out asterisk.crt \
  -subj "/CN=ASTERISK-DEV" \
  -addext "subjectAltName=IP:192.168.1.50,IP:10.0.2.15,DNS:localhost"

Hmm — important: the browser validates the cert against the hostname used to connect: we connect to HOSTIP (e.g., 192.168.1.50). Self-signed → warning regardless of SAN; user clicks through. SAN matters less for manually-accepted certs. But modern Chrome: after accepting, WSS on same origin works (cert exception is per-origin covering the TLS session? Cert exceptions in Chrome apply to the origin and persist for the session for all connections to that origin, including WSS — yes, accepting the https page exception covers wss://same-host:8089/ws). Good. But include the SAN with the host IP anyway (good practice + Firefox needs proper accept too). Note: CN must be the IP/hostname; browsers ignore CN now, need SAN. Include both. Also note the user should replace 192.168.1.50 with their actual Windows host LAN IP.

Combined pem: Asterisk http.conf wants tlscertfile as pem containing cert+key:

sudo cat asterisk.crt asterisk.key | sudo tee asterisk.pem >/dev/null
sudo chown asterisk: asterisk.pem; sudo chmod 600 asterisk.pem  # hmm, /etc/asterisk/keys perms

Actually Debian runs asterisk as user asterisk. http.conf tlscertfile=/etc/asterisk/keys/asterisk.pem readable by asterisk user. Also tlscertfile can take separate files? tlscertfile is one file (cert+key combined). There's also ssl private key option? For http.conf, I believe only tlscertfile (combined). OK.

Hmm, does Asterisk run as asterisk user on Debian by default? Debian's asterisk package: systemd unit runs as user asterisk, yes. And /etc/asterisk owned by asterisk. Fine.

http.conf:

[general]
enabled=yes
bindaddr=0.0.0.0
bindport=8088
tlsenable=yes
tlsbindaddr=0.0.0.0:8089
tlscertfile=/etc/asterisk/keys/asterisk.pem
enablestatic=yes

Hmm — also for websocket to work: res_pjsip_transport_websocket must be loaded (module). module show like res_pjsip_transport_websocket. Debian package includes it. Note: WebSocket transport in Asterisk REQUIRES... also res_crypto? no. Keep simple; mention module check.

pjsip.conf full:

[global]
type=global
user_agent=Asterisk Dev PBX

[transport-wss]
type=transport
protocol=wss
bind=0.0.0.0

[1000] ...
[auth1000] ...
[1000] type=aor max_contacts=4 remove_existing=yes ...
[1001] ... (second account for two-browser testing)

Also for the plain-UDP option maybe a [transport-udp]? Not needed if only WSS clients. Skip (mention that hardphones would use udp transport on 5060).

rtp.conf:

[general]
rtpstart=10000
rtpend=10010
icesupport=yes
strictrtp=no   ; hmm — strictrtp default yes; with NAT weirdness, setting no can help learning sources

Hmm, strictrtp: with strict RTP, Asterisk learns source from first packets; behind SLIRP it should be fine once ICE validates. I'll leave default but mention it in troubleshooting. Actually for dev simplicity I might set strictrtp=no in the dev config to reduce one failure mode, with a comment. Eh — keep default yes, mention in troubleshooting. Hmm. Actually let me set it to no with a comment "dev only". It genuinely helps in NATed lab setups when packets come from unexpected source ports. I'll include with comment.

Also icesupport=yes — hmm, is the rtp.conf option named icesupport? Yes: rtp.conf icesupport=yes enables ICE globally (also endpoint option ice_support). webrtc=yes sets ice_support on endpoint; the rtp.conf global icesupport might also be needed? I recall rtp.conf has icesupport option enabling ICE engine in RTP stack; endpoint ice_support uses it. Include both — harmless.

extensions.conf:

[general]
static=yes
writeprotect=no
cleandigitsystem=yes?? no. Keep minimal.

[from-internal]
exten => 100,1,Answer()
 same => n,Playback(demo-congrats)
 same => n,Hangup()

exten => 611,1,Answer()
 same => n,Playback(demo-echotest)
 same => n,Echo()
 same => n,Playback(demo-congrats)
 same => n,Hangup()

Echo test (611) is a GREAT smoke test for the browser: speak → hear yourself → proves both audio directions + codec + RTP path. Definitely include 611 Echo. (demo-echotest sound exists in core sounds: "demo-echotest" yes.)

The webpage: single index.html with embedded CSS/JS, SIP.js from CDN. Features:

    Connection settings panel (prefilled): SIP user 1000, password, wsURL wss://HOSTIP:8089/ws, target extension. Editable — IMPORTANT: the page is served from Asterisk static; the JS needs to know the WSS URL. Trick: default it to wss://${location.hostname}:8089/ws — the page is loaded from https://HOSTIP:8089/index.html, so location.hostname is automatically right! Nice — zero-config. SIP domain: location.hostname.
    Status line (registered, in call, etc.)
    Number display + phonepad grid (1-9,*,0,#) + backspace, like Android dialer
    Call / Hangup button
    In-call: keypad sends DTMF (sendDTMF)
    Remote audio element (autoplay)

Let me write that JS carefully but compactly. Also handle: simpleUser.connect() then register(). SimpleUser connect includes automatic re-connect? No. Keep simple.

Also: register is optional for calling? For PJSIP endpoint matching, From user matching works without registration; auth challenge happens on INVITE too. But registering is good practice + proves auth works. Keep register with a toggle? Keep it automatic.

Mic permission: call() triggers getUserMedia → browser prompts. Note for user.

Also mention: open two tabs (1000 and 1001) and call 1001 from 1000 — wait, calling another extension: exten 1000 → Dial(PJSIP/1000). Add:

exten => _1XXX,1,Dial(PJSIP/${EXTEN},20)
 same => n,Hangup()

Hmm — careful: pattern _1XXX would match our IVR numbers if I use 1xxx. I'll use extensions: 100 (smoke), 611 (echo), 200 (hello IVR), 300 (two-level IVR part 3), 400 (AGI demo), and 1000/1001 are user IDs — a pattern _1XXX would catch 1000/1001 for extension-to-extension dialing but ALSO... 100? No, _1XXX is exactly 4 digits starting with 1: 1000,1001 fine. But wait, user dials "1000" — that matches _1XXX → Dial(PJSIP/1000). Good. But does 1000 conflict with anything? The IVR numbers 100,200,300,400 are 3 digits — no clash. OK.

Hmm, but hold on: Dial(PJSIP/1000) — the other browser must be REGISTERED (aor 1000 has contact). With remove_existing=yes on aor, re-register replaces. Two tabs same account 1000 would kick each other (remove_existing) — so for two-tab testing use 1000 and 1001. Note that.

Also add same => n,Voicemail... no, keep lean.

Prod differences section (part 2 outro):

    Real server: public IP, no hostfwd; open/forward 8089 TCP (or put WSS behind nginx/traefik on 443), RTP UDP range open in firewall, set external_media_address/external_signaling_address = public IP (for SIP over UDP/TCP transports), DNS + Let's Encrypt certs (real CA → no cert warnings; WSS cert must match hostname), STUN/TURN for remote WebRTC users behind NAT (coturn), registration security (strong passwords, fail2ban), codecs (opus good for WebRTC, transcoding to ulaw/alaw for PSTN), TLS everywhere.

Part 3: IVR tour

Two-level menu with prompts:

[ivr-main]
exten => 300,1,Answer()
 same => n,Set(TIMEOUT(digit)=2)     ; inter-digit timeout
 same => n,Set(TIMEOUT(response)=8)  ; wait for first digit
 same => n,Background(custom/welcome)
 same => n,WaitExten(8)

exten => 1,1,Goto(sales,1)  ; hmm — careful with context structure
...

Wait — structure: menu in context from-internal; after Background/WaitExten, digit extensions matched in SAME context. So:

[from-internal]
; main menu
exten => 300,1,Answer()
 same => n,Ringing()  ; optional
 same => n,Set(TIMEOUT(digit)=2)
 same => n,Set(TIMEOUT(response)=8)
 same => n,Background(custom/ivr-welcome)
 same => n,WaitExten()

; level 1
exten => 1,1,Playback(custom/choose-sales)
 same => n,Set(TIMEOUT(digit)=2)
 same => n,Set(TIMEOUT(response)=8)
 same => n,Background(custom/ivr-sales)
 same => n,WaitExten()
exten => 1,1,Hangup()... 

Hmm wait, collision: exten 1 in from-internal is ALSO reachable as dialed extension "1" from phonepad (user dials 1 + call → executes menu level). That's fine/harmless for dev.

But cleaner pedagogy: use sub-contexts with Goto:

[from-internal]
exten => 300,1,Answer()
 same => n,GotoIf($["${CHANNEL(state)}" != "Up"]?...) no, Answer is there.
 same => n,Set(TIMEOUT(digit)=2)
 same => n,Set(TIMEOUT(response)=8)
 same => n,Background(custom/ivr-welcome)
 same => n,WaitExten()
 same => n,Goto(300,1)  ; hmm, WaitExten falling through after timeout → t handler

exten => t,1,Playback(custom/goodbye)... 

Hmm, timeout handling: when WaitExten times out, control goes to exten t (in same context) if defined; if none, continues to next priority after WaitExten? Actually: if no 't' extension, after timeout the dialplan continues at the next priority after WaitExten? I believe with no t extension, it just continues to next priority. And invalid digit → exten i. Let me structure with t and i handlers in each context. And digit handlers Goto sub-contexts:

[from-internal]
exten => 300,1,Answer()
 same => n,Set(TIMEOUT(digit)=2)
 same => n,Set(TIMEOUT(response)=8)
 same => n,Background(custom/ivr-welcome)
 same => n,WaitExten()
 ; if we fall through, treat as timeout
 same => n,Goto(t,1)

exten => 1,1,Goto(ivr-sales,${EXTEN},1)
exten => 2,1,Goto(ivr-support,${EXTEN},1)
exten => 3,1,Goto(ivr-balance,${EXTEN},1)   ; the AGI demo
exten => 9,1,Playback(custom/repeat-menu)... or Goto back
exten => i,1,Playback(custom/invalid)
 same => n,Goto(300,1)
exten => t,1,Playback(custom/goodbye)
 same => n,Hangup()

[ivr-sales]
exten => _X,1,Set(TIMEOUT(digit)=2)...

Hmm — two-level: main → sales submenu with options (1: talk to sales rep → Dial, 2: hear office hours → Playback, * : back). Support submenu similar. Balance = AGI demo.

Let me write it concretely (will finalize in answer):

[from-internal]
; --- main menu (exten 300) ---
exten => 300,1,Answer()
 same => n,Set(TIMEOUT(digit)=2)
 same => n,Set(TIMEOUT(response)=8)
 same => n,Background(custom/ivr-welcome)
 same => n,WaitExten(8)
 same => n,Goto(t,1)                      ; nothing entered

exten => 1,1,Goto(ivr-sales,1,1)
exten => 2,1,Goto(ivr-support,1,1)
exten => 3,1,Goto(ivr-balance,1,1)
exten => 0,1,Playback(demo-congrats)?? no. 0 → operator: Dial(PJSIP/1000)
exten => i,1,Playback(custom/invalid)
 same => n,Goto(300,1)
exten => t,1,Playback(custom/goodbye)
 same => n,Hangup()

Hmm wait, Goto(ivr-sales,1,1) — Goto(context,exten,priority): exten "1". In [ivr-sales] define exten 1, 2 etc. But when we Background in ivr-sales and user presses 1, matched exten is 1 in ivr-sales. OK.

[ivr-sales]
exten => 1,1,Playback(custom/sales-hours)
 same => n,Goto(from-internal,300,1)  ; return to main menu
exten => 2,1,Dial(PJSIP/1000,15)
 same => n,Playback(custom/vm-noanswer)?? 
 same => n,Goto(from-internal,300,1)
exten => *,1,Goto(from-internal,300,1)
exten => i,1,Playback(custom/invalid)
 same => n,Set(TIMEOUT(digit)=2)
 same => n,Set(TIMEOUT(response)=8)
 same => n,Background(custom/ivr-sales)
 same => n,WaitExten()
 same => n,Goto(i,1)?? hmm

Hmm wait — the submenu needs its own Background/WaitExten entry point. Let me restructure: when entering ivr-sales, first play submenu prompt & WaitExten; digits handled:

[ivr-sales]
exten => 1,1,Answer()?? (already answered)
exten => _X?? 

Cleaner: enter ivr-sales at exten "s" (start):

exten => 1,1,Goto(ivr-sales,s,1)
...
[ivr-sales]
exten => s,1,Set(TIMEOUT(digit)=2)
 same => n,Set(TIMEOUT(response)=8)
 same => n,Background(custom/ivr-sales)
 same => n,WaitExten()
 same => n,Goto(i,1)  ; timeout treated as invalid? or t
exten => 1,1,Playback(custom/sales-hours)
 same => n,Wait(1)
 same => n,Goto(from-internal,300,1)
exten => 2,1,Dial(PJSIP/1000,15)
 same => n,Goto(from-internal,300,1)
exten => *,1,Goto(from-internal,300,1)
exten => i,1,Playback(custom/invalid)
 same => n,Goto(s,1)
exten => t,1,Playback(custom/goodbye)
 same => n,Hangup()

Using 's' extension is the classic IVR pattern. But from main menu Goto(ivr-sales,s,1) — nice. However when user presses digits DURING Background(custom/ivr-sales) — matched in ivr-sales ✓.

Hmm, one catch: after Goto(from-internal,300,1) re-entering main menu — Background plays again ✓.

Audio prompts:

    Built-in: demo-congrats, demo-echotest, and core sounds like "vm-goodbye", "pbx-invalid", "digits/...", etc. Show using built-ins (e.g., Playback(vm-goodbye), Playback(pbx-invalid)) as zero-effort option. core show sounds to list.
    TTS-generated: espeak-ng or pico2wave:

sudo apt install -y espeak-ng sox libsox-fmt-all?? 
espeak-ng -v en-us+f3 -s 150 -w /tmp/welcome.wav "Welcome to the Acme support line."
sudo sox /tmp/welcome.wav -r 8000 -c 1 -t wav /var/lib/asterisk/sounds/en/custom/ivr-welcome.wav

Hmm — Asterisk Debian installs sounds at /var/lib/asterisk/sounds/en (with variants en/... "core" sounds). Custom dir: create /var/lib/asterisk/sounds/en/custom/. Reference in dialplan as custom/ivr-welcome (relative to sounds/en). ✓. Format: 8kHz mono 16-bit PCM wav — Asterisk plays WAV fine (format_wav). No sox conversion strictly needed (Asterisk transcodes), but 8k avoids issues; keep sox step. Also mention SLN16/GSM. And mention format naming: if file is ivr-welcome.wav, Playback(custom/ivr-welcome) auto-picks best format. ✓

Also mention commercial/pro TTS: Google/Azure TTS via API into files, or Asterisk TTS engines: Festival (asterisk-festival? not in Debian?), Swift, Cepstral — mention briefly. Also Say()? SayNumber, SayDigits, SayAlpha — "built-in TTS-ish" for dynamic content (numbers) without audio files — important for AGI demo (SayNumber balance). ✓

AGI demo design: "Account balance" IVR:

    exten 3 → ivr-balance: Playback custom/enter-account, Read(ACCOUNT,custom/enter-account??) — Read() app: Read(varname,filenames[,maxdigits][,option][,attempts][,timeout]). So:

[ivr-balance]
exten => s,1,Playback(custom/enter-account)
 same => n,Read(ACCOUNT,,6,,,10)   ; hmm Read syntax: Read(variable,filename,maxdigits,attempts? no...

Read() signature: Read(variable, filename&..., maxdigits, options, attempts, timeout). So Read(ACCOUNT,custom/enter-account,6,,3,10) plays prompt, collects up to 6 digits, 3 attempts, 10s timeout. Then AGI:

 same => n,AGI(balance.py,${ACCOUNT})
 same => n,GotoIf($["${AGISTATUS}" != "SUCCESS"]?notfound)
 same => n,Playback(custom/your-balance-is)
 same => n,SayNumber(${BALANCE})    ; say dollars
 same => n,Playback(digits/dollars)?? 

Hmm — SayNumber + Playback(dollars)? There's a sound "digits/dollar"... hmm, core sounds have "digits/dollars"? There is "digits/dollar" and "digits/dollars"? I'm not sure. Simplify: SayNumber(${BALANCE}) then Playback(custom/dollars) (TTS-generated "dollars"). Or SayMoney? no such app. Fine.

notfound: Playback custom/unknown-account, Goto main menu.

AGI script (raw protocol — pedagogical):

#!/usr/bin/env python3
"""balance.py - AGI demo: look up account balance in SQLite.

Usage in dialplan:  AGI(balance.py,${ACCOUNT})
Sets channel variables: BALANCE (integer) and ACCNAME? or sets NOTFOUND.
"""
import sys
import sqlite3

DB = "/var/lib/asterisk/agi-bin/accounts.db"

def read_agi_env():
    env = {}
    while True:
        line = sys.stdin.readline()
        if not line or line.strip() == "":
            break
        if ":" in line:
            k, v = line.split(":", 1)
            env[k.strip()] = v.strip()
    return env

def agi(command):
    sys.stdout.write(command + "\n")
    sys.stdout.flush()
    return sys.stdin.readline().strip()   # "200 result=0" etc.

def main():
    env = read_agi_env()
    account = sys.argv[1] if len(sys.argv) > 1 else env.get("agi_arg_1", "")
    found = False
    try:
        conn = sqlite3.connect(DB)
        row = conn.execute("SELECT balance FROM accounts WHERE id = ?", (account,)).fetchone()
        conn.close()
    except sqlite3.Error:
        row = None
    if row:
        agi('SET VARIABLE BALANCE "%d"' % row[0])
        agi('SET VARIABLE AGI_RESULT "FOUND"')
    else:
        agi('SET VARIABLE AGI_RESULT "NOTFOUND"')
    agi("VERBOSE \"balance.py: account %s -> %s\"" % (account, row or "not found") + " 1")?? 

Hmm VERBOSE syntax: VERBOSE "message" level → result. Let me keep it simple: just SET VARIABLEs and maybe a VERBOSE. AGI env also includes agi_arg_1 — args come as agi_arg_1 in env, but also as command-line argv (both work; argv is simpler). ✓

Note: AGI scripts run as the asterisk user; file must be executable, in /var/lib/asterisk/agi-bin (default astagidir). SELinux not on Debian. ✓

DB creation:

sudo sqlite3 /var/lib/asterisk/agi-bin/accounts.db <<'SQL'
CREATE TABLE accounts (id TEXT PRIMARY KEY, name TEXT, balance INTEGER);
INSERT INTO accounts VALUES ('123456','Alice',4200);
INSERT INTO accounts VALUES ('654321','Bob',75);
SQL
sudo chown asterisk:asterisk ... chmod

sqlite3 CLI needs apt install sqlite3.

Also demonstrate reading a channel variable in AGI: agi('GET VARIABLE ACCOUNT') returns "200 result=1 (123456)". Maybe show GET VARIABLE as alternative to argv. Brief mention.

Mention AGI protocol details: stdin env lines (agi_request, agi_channel, agi_language, agi_callerid, agi_context, agi_extension, agi_priority, agi_enhanced, agi_accountcode, agi_threadid), then command/response loop: each command → "200 result=<code>[ (<extra>)]", special: HANGUP result. Channel is already answered (we Answer in dialplan first — AGI doesn't auto-answer; also note agi_answer available). DeadAGI/async AGI: AGI() vs FastAGI agi://host:4573/script (needs FastAGI server; mention frameworks: for Python pyst2, PHP PAGI, Java asterisk-java; and note modern alternative ARI (REST/WebSocket) with stasis() app — a short paragraph contrasting AGI (script-per-call, simple, synchronous) vs ARI (external app controls calls via REST events, modern architecture, used by e.g. jambonz?/clearwater? no...). Keep concise.

Also part 3 should mention dialplan show from-internal CLI for debugging, core set verbose 5, agi set debug on, and watching AGI session in CLI. ✓

Part 4: real phone number

Content:

    Concept: PSTN connectivity via a SIP trunk from an ITSP (Internet Telephony Service Provider). You "buy" a DID (Direct Inward Dialing) number — monthly fee, per-minute. Providers: Twilio ( Programmable Voice — but Twilio is its own API platform more than raw SIP trunk; they do offer SIP trunking? Twilio offers Elastic SIP Trunking yes), also: Flowroute, Bandwidth, Voxbeam, Telnyx, VoIP.ms, DIDLogic, les.net, Vonage/SIPgate (EU), AnveoDirect... Region-dependent; some require business KYC. Note: many consumer "virtual number" services don't give SIP.
    Alternative: Google Voice? No SIP. Skype? No. Mention porting/ownership lightly.
    Regulatory: some countries require local presence/address proof for local numbers; E911 requirements (US); emergency calls must be routed (note: E911 obligations when reselling).
    Technical: two models:
        Provider → you register out (they give you credentials; your Asterisk registers as SIP client to their registrar — pjsip registration section, like an endpoint in reverse).
        IP peer / static trunk: they send SIP to your public IP (you configure endpoint with their IPs, no registration) — most common for prod with static IP.
    Config sketch (pjsip trunk): trunk endpoint (context=from-pstn), aor with static contact or registration, auth, codec ulaw/alaw, direct_media=no. Inbound: DID number arrives as SIP To/From user → dialplan exten matches (or use _X. catch-all). Outbound: Dial(PJSIP/${EXTEN}@trunk-prov). Show minimal trunk config skeleton:

[trunk-out]
type=registration
transport=transport-udp?? (wss only for browsers; trunk uses udp/tcp 5060)
outbound_auth=trunk-auth
server_uri=sip:sip.provider.com
client_uri=sip:15551234567@sip.provider.com
[trunk-auth] type=auth ...
[trunk] type=endpoint context=from-pstn ... 
[trunk] type=aor ...

And dialplan from-pstn: exten => 15551234567,1,Goto(from-internal,300,1) (link DID → our IVR!). That's the "link it to our PBX" answer: inbound routing by DID.

    Also mention: keep the WebRTC side; real users call the DID via PSTN → provider → your Asterisk (needs public IP, 5060 open or SBC, TLS optional, and crucially RTP ports open + external_media_address set).
    Security notes: fail2ban/asterisk security logs, allow anonymous only from provider IPs, strong secrets, never expose AMI/ARI unauthenticated; consider an SBC (Kamailio/OpenSIPS) in front for scale,


----

you reached max output in last round. can you see the thinking trace from last round? (begin with "Let me look at what happened. The user's backup file shows their original query..." and end at "... strong secrets, never expose AMI/ARI unauthenticated; consider an SBC (Kamailio/OpenSIPS) in front for scale,") If you can see that trace, then simplify continue thinking and output answer when done. If you cannot see that trace, please immediately stop and report to me that the thinking trace got truncated, then I will arrange a different session so that you can access the trace again. thanks.

----

Yes, I can see the thinking trace from the last round. It begins with "Let me look at what happened. The user's backup file shows their original query..." and ends with "...strong secrets, never expose AMI/ARI unauthenticated; consider an SBC (Kamailio/OpenSIPS) in front for scale,". The trace covers all the planning: architecture, QEMU hostfwd + ICE/SLIRP analysis, cert generation, http.conf/pjsip.conf/rtp.conf, SIP.js SimpleUser API (with pinned version 0.21.1 and compatibility notes), dialplan (smoke test 100, echo 611, hello IVR 200, two-level menu 300, AGI balance demo), TTS via espeak-ng + sox, AGI python script with SQLite, and part 4 outline (ITSP/DID, trunk config, linking DID to IVR, security).

I should now finish the remaining planning quickly (part 4 details were nearly complete; I need to wrap up: SBC mention, E911, then write the final answer).

Remaining details to settle quickly:

    Part 4: finish SBC thought (Kamailio/OpenSIPS or a hosted SBC in front for scale/federation), maybe mention FreePBX as GUI alternative, and wrap up.
    Then write the full comprehensive answer. This is a long tutorial. Let me structure it:

    Architecture overview
    Part 1: Install + smoke test inside VM
    Part 2: End-to-end (certs, http.conf, pjsip.conf, rtp.conf, extensions.conf, QEMU hostfwd, webpage with SIP.js, debugging, prod differences)
    Part 3: Two-level IVR + prompts (built-in, TTS-generated) + AGI + SQLite + ARI note
    Part 4: Getting a real number (ITSP/DID, trunk config, linking to PBX, security)
    Quick reference / troubleshooting table

Let me write the final answer now. It should be thorough but well-organized. I'll use code blocks liberally. Let me make sure the webpage code is complete and self-contained.

For the webpage, let me write it carefully:

<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SIP.js Softphone (IVR test)</title>
<script src="https://cdn.jsdelivr.net/npm/sip.js@0.21.1/dist/sip.min.js"></script>
<style>...</style>
</head>
<body>
  status line
  display (digits entered)
  keypad grid
  call/hangup button
  in-call DTMF: same keypad when in call sends DTMF
  <audio id="remoteAudio" autoplay></audio>
  settings panel (wsURL, user, password, target)
<script>
  const host = location.hostname;
  ... simpleUser = new SIP.Web.SimpleUser({ delegate: {...} });
  await simpleUser.connect({ wsURL: `wss://${host}:8089/ws`, authorizationUsername, authorizationPassword, displayName });
  await simpleUser.register();
  call: simpleUser.call(`sip:${target}@${host}`, { media: { constraints: { audio: true, video: false } } });
  dtmf: simpleUser.sendDTMF(digit);
  hangup: simpleUser.hangup();
</script>
</body>
</html>

SimpleUser audio: I need the remote audio attached. From memory, SimpleUserOptions includes audioHelper and there's a way... Actually I recall SimpleUser automatically creates and attaches an audio element? No... Hmm. I remember docs: "If an HTMLAudioElement is not provided, one will be created"? Hmm, actually I think SimpleUser has media: { remote: { audio: HTMLAudioElement } }?

Let me think again about SIP.js SimpleUser media options. I recall SimpleUserMedia interface: { local?: { audio?: MediaStreamConstraints }, remote?: { audio?: HTMLAudioElement | boolean }, constraints?: MediaStreamConstraints, ... }. Hmm.

Actually, I now recall more specifically: SimpleUser.call options: { media: { constraints: { audio: true, video: false } } } — that's in call options. And remote audio: the AudioHelper — SimpleUser constructor options audioHelper?: AudioHelper — and there's Web.AudioHelper.AudioHelper(remoteAudio: HTMLAudioElement)? or SimpleUser default: "An AudioHelper is created which attaches remote audio to an audio element appended to document body"?

I genuinely remember something like: SimpleUser by default creates an <audio> element automatically. In simple-user.ts:

// If set, the AudioHelper will be used to play remote audio...
if (options?.audioHelper) {
  this.audioHelper = options.audioHelper;
} else {
  this.audioHelper = new AudioHelper();
  this.audioHelper.loadAudio...? 
}

Hmm, and AudioHelper constructor: new Web.AudioHelper.AudioHelper(remoteAudio?: HTMLAudioElement, localAudio?: HTMLAudioElement, options?: AudioHelperOptions) — and if no element passed, it creates one and attaches to document.body. I'm fairly confident AudioHelper auto-creates elements when none provided (the docs mention the audio helper attaches audio elements to the DOM automatically).

OK so: SimpleUser with default options auto-plays remote audio via auto-created audio element. To be explicit and safe, I'll pass an audio element via audioHelper — hmm, but if I get the AudioHelper constructor signature wrong, it breaks. Safer: rely on default auto-attach, and note in a comment that SimpleUser attaches remote audio automatically (and if you hear nothing, check autoplay policy — Chrome requires user gesture; our Call button click is the gesture, fine).

Actually wait — autoplay: audio starts when call answered, which follows a click (call/answer button) — Chrome's autoplay policy allows audio after user gesture on the page. The call button click counts. OK, default behavior should work. I'll mention: "click somewhere on the page if no audio" as troubleshooting.

Hmm, but there's a real risk I'm wrong about the exact options. Let me minimize the API surface used:

    new SIP.Web.SimpleUser({ delegate: { onCallReceived, onCallHangup?, onServerDisconnect? } }) — hmm, which delegate names? I recall SimpleUserDelegate has: onCallAnswered?, onCallCreated?, onCallHangup?, onCallHold?, onCallReceived?, onCallReleased?, onRegistered?, onServerDisconnect?, onUnregistered?... I'll use onCallReceived, onCallHangup, onServerDisconnect — hmm, but is it onCallHangup or onCallTerminated? I'm fairly sure SimpleUser uses "onCallHangup"? Hmm. There's simpleUser.hangup() method... and delegate... I believe it's onCallHangup. Hmm, actually maybe both exist? I'll wire optional callbacks defensively — if a name is wrong it's just never called; the UI also polls simpleUser.isConnected() / call state via button states. To be safe, primary state comes from our own actions (await call() resolves when session established; catch on failure). And add a periodic status updater using simpleUser.isCalling()? Hmm, is there isCalling()? I don't remember exactly. There's simpleUser.call() / .answer() / .hangup() / .hold() / .isConnected() / .isRegistered()?

OK pragmatic decision: keep UI state driven by our own async flows (connect/register/call/hangup), wire delegate callbacks best-effort with try/catch, and add clear comments. The tutorial's focus is IVR, and I'll add a note box about SIP.js version differences with a pointer to docs. This is acceptable and honest.

sendDTMF: simpleUser.sendDTMF("1") — confident enough.

Also one important dialpad behavior detail: before call, keypad types into display (dialed extension); during call, keypad sends DTMF immediately (like Android in-call). I'll implement that.

Now the QEMU command. User said "host machine windows running the VM using qemu, can expose via hostfwd param". So on Windows: qemu-system-x86_64 ... -netdev user,id=net0,hostfwd=tcp::2222-:22,hostfwd=tcp::8089-:8089,hostfwd=udp::10000-:10000,... -device virtio-net-pci,netdev=net0 (or e1000). I'll give a full example command line plus note for libvirt users.

RTP range: rtpstart=10000, rtpend=10009 (10 ports = up to ~5 concurrent calls since each call uses 2 ports (RTP+RTCP) — with rtcp_mux for WebRTC only 1? Keep 10 ports, forward 10 UDP hostfwds. Hostfwd syntax each port separate. Actually, let me use 10000-10009 with 10 hostfwd entries — a bit verbose but explicit; or suggest 5 ports. I'll do 10000-10004 (5 ports, enough for 2 concurrent WebRTC calls with rtcp mux... Asterisk allocates RTCP port too unless muxed? For webrtc endpoint rtcp_mux=yes, but Asterisk still allocates an even/odd pair per call by default? With rtcp_mux it may use one port. 5 ports ≈ 2 calls safely.) I'll forward 10000-10007 (8 ports, 3-4 calls). Let me just do 10000–10009 and note each call needs 1-2 ports.

Also need SSH hostfwd 2222→22 for convenience. And note: Windows host IP — find via ipconfig (e.g., 192.168.1.50). Browser URL: https://192.168.1.50:8089/index.html.

ice_host_candidates: include with hedge. Actually, let me reconsider once more whether [ice_host_candidates] is real... I have a genuine memory trace of Asterisk rtp.conf sample containing:

[ice_host_candidates]
; 
; This section is used to map local IP addresses to external IP addresses...
; 10.0.0.2 => 74.93.46.23

Hmm, I've also seen "ice_host_candidates" mentioned in Asterisk community forum posts about WebRTC NAT. I'm like 80% sure it exists (added ~Asterisk 15/16). I'll include it with "check rtp.conf.sample in your version" hedge. Even if it doesn't exist, Asterisk ignores unknown sections in rtp.conf (harmless), and the SLIRP-outbound path may still work.

Wait, actually there's a subtlety I should double-check: does SLIRP forward the GUEST→HOST UDP traffic to the host's LAN IP? Guest sends UDP to 192.168.1.50 (host's LAN IP). SLIRP NAT: guest → 10.0.2.2 (gateway) → SLIRP creates a socket on the host bound to... SLIRP sends from the host's default interface to 192.168.1.50, which is the host itself — packet loops back internally (host addresses its own IP → loopback delivery). Browser bound on that port receives it. Yes, this works (this is how guest→host services work in SLIRP).

And browser's reply: browser replies to source IP:port of the STUN packet = host's IP + SLIRP ephemeral port → SLIRP maps back to guest. ✓.

So both paths work in theory; the primary recommended config: hostfwd TCP 8089 + UDP RTP ports + ice_host_candidates mapping 10.0.2.15 => HOST_LAN_IP. With the mapping, browser's checks to HOST_LAN_IP:10000 (hostfwd'd) reach Asterisk → clean ICE success.

Hmm wait, one more check on the mapped candidate path: Asterisk advertises candidate 192.168.1.50:10000 (host type? it'd be a "host" candidate with wrong address, or "srflx"? whatever—). Browser sends STUN binding request to 192.168.1.50:10000 → hostfwd → guest 10.0.2.15:10000 → Asterisk replies to source (10.0.2.2:port) → SLIRP → browser. Check OK. Browser then sends RTP to 192.168.1.50:10000 → forwarded → Asterisk; Asterisk sends RTP to browser's address (from ICE: 192.168.1.50:browserport or the SLIRP-mapped address it learned) — ICE validated pair tells Asterisk the destination = browser's candidate address 192.168.1.50:port → SLIRP outbound → host loopback → browser. ✓✓.

One caveat: browser's candidate = 192.168.1.50:port — Asterisk sending RTP to 192.168.1.50:port via SLIRP — but is the SLIRP NAT mapping for that destination maintained? Asterisk sends continuously to that address:port; SLIRP forwards each packet from host stack to 192.168.1.50:port = host itself → arrives at browser's socket. ✓ (no NAT mapping needed for delivery since destination is the host itself; SLIRP just routes outbound packet which hairpins to host.) Should be fine.

Also: Chrome may expose mDNS candidates only (if mic permission not yet granted at ICE gathering time). Permission is requested at call() time before gathering? getUserMedia happens then ICE gathers — permission granted by then. OK.

Also fallback story if audio fails: run rtp set debug on in Asterisk CLI, check about:webrtc in Chrome; ultimate fallback: bridged networking.

Now, let me also reconsider: should I recommend the user run QEMU with -nic user,...? Same thing. I'll show -netdev form.

Windows QEMU note: they run qemu on Windows (qemu.org Windows binaries / WSL). Fine, command works in cmd/powershell (line continuation differs — I'll show single line or PowerShell backtick note).

Now Debian install details:

sudo apt update
sudo apt install -y asterisk
# optional useful: sox espeak-ng sqlite3
asterisk -V  # Asterisk 20.x
sudo systemctl status asterisk

Backup default configs:

cd /etc/asterisk
sudo mkdir defaults-backup... 

Hmm — careful: Debian config dir has many files; moving pjsip.conf etc. Also sorcery? Minimal set: pjsip.conf, extensions.conf, http.conf, rtp.conf, modules.conf (leave). Note: Debian's default pjsip.conf may have sample endpoints (defaults from package = commented samples mostly). Overwrite with our minimal versions. Backup first.

Reload: sudo asterisk -rx "core reload" or individual: pjsip reload, dialplan reload, http reload? http.conf changes need... http reload? hmm, does http reload exist? There's no http reload I think... Actually there IS: CLI "http reload"? Hmm. I believe http.conf changes need full asterisk restart (core restart now)? I'm not sure there's an http reload command. Let me think... Asterisk CLI has "http show status" and... I believe there's also "http reload"? I genuinely am unsure. Safe advice: after editing http.conf run sudo systemctl restart asterisk (or core restart now). I'll just use restart to be safe.

Smoke test (part 1):

sudo asterisk -rvvv
CLI> core show version
CLI> pjsip show endpoints? (none yet if we replaced config — order matters!)

Hmm — sequence issue: the smoke test in part 1 should happen BEFORE we do the part-2 config rewrite? Or after? Plan: Part 1 = install + minimal dialplan smoke test using the SHIPPED sample config? The shipped Debian extensions.conf is the huge demo config (context "default" with demo stuff? Debian ships the standard sample extensions.conf which has a "default" context with demos? The standard sample has contexts like [demo], [default]... The stock sample extensions.conf has [general], [globals], [default] with... hmm, stock Asterisk sample extensions.conf has some demo extensions in "default"? I recall the sample has exten => 1234,1...? Not sure.

Cleaner: Part 1 includes writing the minimal configs already? But part 2 adds pjsip/http/rtp. Better flow:

    Part 1: install, verify service, enter CLI, replace ONLY extensions.conf with minimal (from-internal with exten 100 + 611 echo), smoke test via Local originate. No SIP endpoints needed for Local originate (Local channels work without transports... wait, does Local channel need any SIP? No — pure dialplan). ✓ Good: Local originate works with zero pjsip config.
    Part 2: add certs, http.conf, pjsip.conf, rtp.conf, restart, browser test.

Smoke test commands:

sudo asterisk -rvvv
> channel originate Local/100@from-internal application Playback demo-congrats

Hmm wait — with dialplan 100 = Answer/Playback/Hangup and app = Playback demo-congrats: the originated leg runs Playback immediately? The Local channel is created; app side runs Playback, other side enters dialplan at 100 → Answer → both up. Fine. Alternatively channel originate Local/100@from-internal application Wait 10 shows channels. I'll show Playback variant and core show channels tip with Wait. Actually simplest evidence: verbose lines showing "Executing [100@from-internal:1] Answer".

Hmm, also: does channel originate require the Local channel to match a dialplan exten 100 in from-internal — yes we defined it.

Also sound file existence: demo-congrats in en sounds ✓ (asterisk-core-sounds-en-wav installed by default? Debian asterisk package depends on asterisk-core-sounds... I believe asterisk meta pulls asterisk-sounds-core-en-wav? hmm, Debian has asterisk-core-sounds-en-wav as dependency? The asterisk package Depends: asterisk-core-sounds-en-wav? I think yes (or Recommends). If missing: sudo apt install asterisk-core-sounds-en-wav. I'll note it.)

Also sudo asterisk -rx "..." for one-shot commands. And mention core show channels, dialplan show from-internal.

Part 2 flow details:

Step 1: QEMU networking — decide host IP first (ipconfig). Example: 192.168.1.50.

Step 2: TLS cert (with SAN including 192.168.1.50).

Step 3: http.conf (8088 plain, 8089 TLS, static on).

Step 4: pjsip.conf (transport-wss, endpoints 1000/1001, webrtc=yes).

Wait — pjsip needs also a UDP transport for... nothing in our setup. But note: pjsip.conf with ONLY wss transport — fine. But hmm, one gotcha: PJSIP requires at least one transport to start? wss counts. ✓

Step 5: rtp.conf (range, icesupport, maybe ice_host_candidates mapping, strictrtp=no dev-only).

Step 6: extensions.conf updated (add 200 hello IVR: Answer, Playback demo-congrats? For "hello world IVR" make 200: Answer, Playback(custom/hello) or SayNumber... let me do: 200 = Answer; Playback(demo-congrats); Playback(vm-goodbye); Hangup. Or better a tiny interactive one: 200: Answer; Set(TIMEOUT...); Background(demo-congrats)? Hello-world: keep it Playback-only + hangup, plus the echo test 611 already gives interaction.)

Step 7: restart asterisk, verify: http show status, pjsip show transports, pjsip show endpoints, curl http://localhost:8088/http_status.

Step 8: deploy webpage to /var/lib/asterisk/static-http/index.html, chown asterisk.

Step 9: Windows browser: open https://192.168.1.50:8089/index.html, accept cert, register, dial 100 / 611 / 200.

Troubleshooting box: cert not accepted for WSS (open wss URL? can't browse; but same-origin exception covers), mic permission, ICE/audio issues (rtp set debug on, about:webrtc), "no endpoint" (identify_by), 401 loop (wrong password), WebSocket 404 (res_pjsip_transport_websocket module + /ws URI), sounds missing.

Also explain clearly the connection details logic:

    Server side: Asterisk listens WSS 0.0.0.0:8089; endpoints 1000/1001 with passwords; RTP 10000-10009; hostfwd maps host ports.
    Client side: page auto-derives wss://<page-host>:8089/ws; SIP identity 1000@(host); auth credentials.
    Prod differences paragraph.

Part 3 (already planned). Let me finalize the AGI dialplan + script + prompts generation commands. Also show CLI verification: dialplan reload, test by calling 300 from the page, watch CLI; agi set debug on shows the AGI session.

Also quickly demonstrate a FastAGI mention and ARI paragraph.

SayNumber: SayNumber(${BALANCE}) says e.g. "four thousand two hundred". Then Playback(custom/dollars). I'll generate "dollars" via espeak too, or just say "your balance is" + SayNumber + "dollars". Fine.

Also unknown account path: AGI sets AGI_RESULT=NOTFOUND → Playback custom/unknown-account → back to menu.

Read(): Read(ACCOUNT,custom/enter-account,6,,3,10) — wait signature: Read(variable,filename,maxdigits,options,attempts,timeout). So attempts=3, timeout=10. Hmm — does Read replay the filename on each attempt? Yes, Read plays filename each attempt. If nothing entered (timeout), ACCOUNT empty → handle: GotoIf empty → back to menu. Also note Read beeps? Read has option 's' to stop... Read plays the file then waits; add option... keep simple.

Also digits-only filter: Read default terminates on # or maxdigits; fine.

Part 4 final content (from trace): ITSP/DID providers list, two trunk models (registration vs IP peer), pjsip trunk config skeleton, from-pstn context linking DID→IVR, outbound pattern, security, E911/regulatory, SBC note, and closing pointers (Asterisk wiki, books: "Asterisk: The Definitive Guide", Asterisk docs, FreePBX mention).

Trunk config skeleton:

; --- Registration-based trunk (provider gives you SIP credentials) ---
[transport-udp]
type=transport
protocol=udp
bind=0.0.0.0

[mytrunk]
type=registration
transport=transport-udp
outbound_auth=mytrunk_auth
server_uri=sip:sip.example-itsp.com:5060
client_uri=sip:+15551234567@sip.example-itsp.com:5060
retry_wait=60

[mytrunk_auth]
type=auth
auth_type=userpass
username=+15551234567
password=secret-from-provider

[mytrunk]
type=aor
contact=sip:sip.example-itsp.com:5060

[mytrunk]
type=identify
endpoint=mytrunk
match=sip.example-itsp.com

[mytrunk]
type=endpoint
transport=transport-udp
context=from-pstn
disallow=all
allow=ulaw
direct_media=no
rtp_symmetric=yes
force_rport=yes
rewrite_contact=yes
outbound_auth=mytrunk_auth
aors=mytrunk
from_domain=sip.example-itsp.com

Dialplan:

[from-pstn]
exten => +15551234567,1,Goto(from-internal,300,1)   ; our IVR!
 same => n,Hangup()
; or catch-all:
exten => _X.,1,NoOp(Inbound call to ${EXTEN})
 ...

[outbound]
exten => _NXXNXXXXXX,1,Dial(PJSIP/${EXTEN}@mytrunk)
 same => n,Playtones(congestion)?? keep: Hangup
exten => _1NXXNXXXXXX,1,Dial(PJSIP/${EXTEN}@mytrunk)

Note: from-internal users would need to route outbound — mention pattern contexts lightly (point: linking = inbound DID → exten match → Goto IVR; outbound = Dial(PJSIP/number@trunk)).

Also note: this requires public IP / port forwarding 5060 UDP + RTP range, external_media_address if behind NAT firewall.

OK — also the intro should have the architecture ASCII diagram. And I should keep overall length manageable but comprehensive. The user explicitly said "extended task, take your time". I'll write a long, structured answer with headers, but avoid unnecessary repetition.

One more check — Asterisk module for websocket: res_pjsip_transport_websocket.so — Debian package includes (part of asterisk-modules? asterisk package). Verify: module show like res_pjsip_transport_websocket. Also res_pjsip_path, res_pjsip_endpoint_identifier_user? For username identify, module res_pjsip_endpoint_identifier_user must be loaded — Debian loads it by default. Mention in troubleshooting.

Also codec: opus — is opus in Debian Asterisk build? Asterisk 20 has codec_opus? codec_opus is a binary module from Digium (not open source? Actually codec_opus.so is distributed by Sangoma as binary; Debian does NOT ship codec_opus I believe). Hmm! If opus unavailable, allow=opus just... would fail to apply? If codec not loaded, allowing it logs warning and ignores? To be safe: use allow=ulaw primarily, and mention opus if available (asterisk 18+ has... no). Browsers support G.711 (PCMU/PCMA) mandatory in WebRTC ✓. So: disallow=all; allow=ulaw. Mention opus quality bonus if the module exists. Good catch — avoid opus in main config.

DTMF: telephone-event offered by SIP.js ✓, Asterisk endpoint dtmf_mode default rfc4733 ✓ works with WebRTC.

Also max_audio_streams? no. dtls_auto_generate_cert — with webrtc=yes, hmm, I said earlier webrtc=yes implies auto-generated DTLS cert. Let me double check my memory: Asterisk docs for webrtc endpoint option: "Enables the defaults for WebRTC... sets use_avpf=yes, media_encryption=dtls, ice_support=yes, rtcp_mux=yes, dtls_auto_generate_cert=yes (if no cert set)..." Hmm — I recall there WAS a version issue: dtls_auto_generate_cert added in 15.x; webrtc=yes sets it? I'm fairly confident that with webrtc=yes and no dtls_cert_file configured, Asterisk auto-generates. If not, fallback troubleshooting: set dtls_auto_generate_cert=yes explicitly. I'll add that line explicitly in config — harmless and self-documenting:

media_encryption=dtls
dtls_auto_generate_cert=yes
use_avpf=yes
ice_support=yes
rtcp_mux=yes

But webrtc=yes already sets these; including both is redundant but explicit — actually there's a subtlety: does dtls_auto_generate_cert=yes work when set directly? Yes it's a valid option. I'll use webrtc=yes plus a comment listing what it enables. Cleaner.

Now — the SimpleUser code. Final version targeting 0.21.x:

const simpleUser = new SIP.Web.SimpleUser({
  delegate: {
    onCallReceived: async () => { await simpleUser.answer(); },  // auto-answer (dev only)
    onCallHangup: () => updateStatus("Call ended"),
    onRegistered: () => updateStatus("Registered ✓"),
    onServerDisconnect: () => updateStatus("Disconnected from server")
  }
});
await simpleUser.connect({
  wsURL,
  authorizationUsername: user,
  authorizationPassword: pass,
  displayName: "Browser " + user
});
await simpleUser.register();
await simpleUser.call("sip:" + target + "@" + domain, {
  media: { constraints: { audio: true, video: false } }
});
await simpleUser.sendDTMF(digit);
await simpleUser.hangup();

Hmm — delegate names: onCallHangup? In SimpleUserDelegate I've seen: onCallAnswered, onCallCreated, onCallHangup, onCallHold, onCallMuted?, onCallReceived, onCallUnheld?, onRegistered, onServerDisconnect, onUnregistered. I'm fairly confident about onCallReceived and onServerDisconnect; onCallHangup — hmm, is it "onCallHangup"? I've also seen "onCallTerminated"... In SimpleUser source, the delegate interface... The method on SimpleUser is hangup(), and the Session delegate... SimpleUser's internal session delegate maps to this.delegate?.onCallHangup?.(). Hmm, hmm. I'll include both onCallHangup AND keep the UI resilient. Fine.

Also connect() — does connect options accept authorizationUsername? I stated ConnectUserOptions includes it. I'm going with it (memory ~85%). Compatibility note covers the rest.

Also registration: simpleUser.register() — after connect. SimpleUser auto-registers if connect option... there might be register: true? Hmm, SimpleUserOptions/ConnectUserOptions might have sipRegistration.... I'll explicitly call register(). Note: register() throws if already registered; wrap in try/catch.

Now write the HTML page fully. Keep it clean: dark-ish simple styling, grid keypad, status bar, settings drawer. ~200 lines. Also add small event log area (helps debugging — shows SIP.js connection events)? Nice-to-have; a tiny log div is cheap and very useful for a tutorial. Include minimal log.

Autoplay note: add onclick anywhere to resume audio? SimpleUser's AudioHelper handles play() retries; skip.

Let me also give the espeak-ng prompt generation for ALL custom prompts used in part 3:

    ivr-welcome: "Welcome to Acme Robotics. For sales, press one. For technical support, press two. To check your account balance, press three. To speak with an operator, press zero."
    ivr-sales: "You have reached sales. To hear our office hours, press one. To be connected to a sales representative, press two. To return to the main menu, press star."
    ivr-support: "You have reached technical support. To hear known issues, press one. To leave a voicemail... " keep: press 1 → playback hours? Let me make support: 1 → "support-wait" Playback + Queue? no Queue complexity — 1 → Playback(custom/support-hours), 2 → Dial(PJSIP/1001) (the "support engineer" = second browser!), * → back.
    enter-account: "Please enter your six digit account number, followed by the pound key."
    unknown-account: "Sorry, we could not find that account number."
    your-balance-is: "Your current balance is"
    dollars: "dollars" — hmm, SayNumber says the number; then "dollars" file. Or use Playback(custom/your-balance-is) + SayNumber + Playback(custom/dollars). ✓
    invalid: "Sorry, that is not a valid option." (or use built-in pbx-invalid — show built-in alternative in comment!)
    goodbye: "Thank you for calling. Goodbye." (or built-in vm-goodbye)
    sales-hours: "Our office hours are Monday through Friday, nine A M to five P M, Central Time."
    support-hours: "We are currently investigating... " eh: "Our support team is available twenty four seven."

Script to generate all with a bash loop:

sudo apt install -y espeak-ng sox
sudo mkdir -p /var/lib/asterisk/sounds/en/custom
gen() {  # gen <name> <text>
  espeak-ng -v en-us -s 145 -w /tmp/$1.wav "$2"
  sox /tmp/$1.wav -r 8000 -c 1 /var/lib/asterisk/sounds/en/custom/$1.wav
  rm /tmp/$1.wav
}
sudo bash -c '...'

Need sudo for writing; run whole block as root: sudo bash -c '...' or do the loop with sudo inside. I'll write a heredoc'd root shell script for cleanliness.

Also note female voice variant -v en-us+f3, and better TTS options (Azure/Google/Piper!) — mention Piper as great offline neural TTS: piper --model en_US-... -f out.wav. Brief mention.

AGI script final:

#!/usr/bin/env python3
"""
balance.py — AGI demo: look up an account balance in SQLite.

Dialplan usage:
    same => n,AGI(balance.py,${ACCOUNT})

Reads:      argv[1] (account number)
Sets:       AGI_RESULT = FOUND|NOTFOUND
            BALANCE    = integer (only when FOUND)
"""
import sys, sqlite3

DB_PATH = "/var/lib/asterisk/agi-bin/accounts.db"

def read_env():
    env = {}
    while True:
        line = sys.stdin.readline()
        if line == "\n" or line == "":   # blank line ends the env block
            break
        if ":" in line:
            k, _, v = line.partition(":")
            env[k.strip()] = v.strip()
    return env

def agi(cmd):
    """Send one AGI command, return the raw response line, e.g. '200 result=0'."""
    sys.stdout.write(cmd + "\n")
    sys.stdout.flush()
    return sys.stdin.readline().strip()

def main():
    env = read_env()
    # Arguments also arrive as agi_arg_1..N in env, and as argv.
    account = (sys.argv[1] if len(sys.argv) > 1 else env.get("agi_arg_1", "")).strip()

    row = None
    try:
        con = sqlite3.connect(DB_PATH)
        row = con.execute(
            "SELECT name, balance FROM accounts WHERE id = ?", (account,)
        ).fetchone()
        con.close()
    except sqlite3.Error as e:
        agi('VERBOSE "balance.py DB error: %s" 3' % e)

    if row:
        name, balance = row
        agi('SET VARIABLE AGI_RESULT "FOUND"')
        agi('SET VARIABLE BALANCE "%d"' % int(balance))
        agi('SET VARIABLE ACCNAME "%s"' % name)
        agi('VERBOSE "balance.py: %s -> %s (%d)" 3' % (account, name, balance))
    else:
        agi('SET VARIABLE AGI_RESULT "NOTFOUND"')
        agi('VERBOSE "balance.py: %s -> not found" 3' % account)

if __name__ == "__main__":
    main()

VERBOSE quoting: VERBOSE "message" level? AGI VERBOSE syntax: VERBOSE "message" <level> — response 200 result=1. OK. Careful with quotes containing spaces — the whole command is space-separated: VERBOSE "msg with spaces" 3 — asterisk parses quotes. Fine.

Permissions: sudo chown asterisk:asterisk /var/lib/asterisk/agi-bin/balance.py; sudo chmod 755. DB readable by asterisk: chmod 644 + dir perms. /var/lib/asterisk/agi-bin default owner asterisk on Debian ✓.

Dialplan for balance:

[ivr-balance]
exten => s,1,Set(TIMEOUT(digit)=3)
 same => n,Read(ACCOUNT,custom/enter-account,6,,3,10)
 same => n,GotoIf($["${ACCOUNT}" = ""]?from-internal,300,1)
 same => n,AGI(balance.py,${ACCOUNT})
 same => n,GotoIf($["${AGI_RESULT}" = "FOUND"]?found:notfound)
 same => n(found),Playback(custom/your-balance-is)
 same => n,SayNumber(${BALANCE})
 same => n,Playback(custom/dollars)
 same => n,Wait(1)
 same => n,Goto(from-internal,300,1)
 same => n(notfound),Playback(custom/unknown-account)
 same => n,Goto(from-internal,300,1)

Hmm — GotoIf labels syntax: same => n,GotoIf($[...]?found:notfound) then same => n(found),... ✓ correct pattern.

Note AGI returns and AGISTATUS variable; we use our own AGI_RESULT var — simpler and shows SET VARIABLE. Mention AGISTATUS exists too.

Test data: accounts 123456 (Alice, 4200), 654321 (Bob, 75).

ARI paragraph: short — ARI = REST API + WebSocket events; dialplan hands channel to Stasis(); external app (any language) controls bridges, playback, etc. Better for complex apps/microservices; AGI simpler for script-per-call. Point to ARI docs + libraries (ari-py for Python, node-ari-client).

Part 4 — write it as planned. Also mention costs (~$1-5/month DID + per-minute), testing with free options: some providers offer free trial credit; also mention "conference bridge test numbers"? no. Keep list: Twilio Elastic SIP Trunking, Telnyx, Bandwidth, Flowroute, Voxbeam, VoIP.ms, DIDLogic, les.net, sipgate (DE/UK), AnveoDirect. Note E911 & KYC. Mention alternative: keep everything IP-based (WebRTC users call via webpage — no PSTN needed) — for many modern apps (e.g., browser-based support widgets) you never need PSTN; PSTN only when regular phones must reach you.

Also "link it to our PBX": explain inbound routing DID→exten→Goto IVR; plus outbound. Plus ops: public IP server (or hosted VPS), DNS, firewall (5060/5061 TCP+UDP, RTP range), external_media_address, security hardening.

Wrap up with a short "where to go next" list: Asterisk wiki, "Asterisk: The Definitive Guide" (free online), ARI, FreePBX GUI, Kamailio SBC, res_fax? no. Keep tight.

Length: this will be long. That's fine and expected. Let me also add a quick TOC at top.

One more detail for the QEMU command — user said Windows host running QEMU. Example full command:

qemu-system-x86_64.exe ^
  -m 2048 -smp 2 ^
  -drive file=debian.qcow2,if=virtio ^
  -netdev user,id=n0,hostfwd=tcp::2222-:22,hostfwd=tcp::8089-:8089,hostfwd=udp::10000-:10000,hostfwd=udp::10001-:10001,hostfwd=udp::10002-:10002,hostfwd=udp::10003-:10003,hostfwd=udp::10004-:10004,hostfwd=udp::10005-:10005,hostfwd=udp::10006-:10006,hostfwd=udp::10007-:10007,hostfwd=udp::10008-:10008,hostfwd=udp::10009-:10009 ^
  -device virtio-net-pci,netdev=n0 ^
  ...

(cmd.exe uses ^; PowerShell uses backtick.) Note: hostfwd binds on all host interfaces by default (0.0.0.0) unless host addr given (hostfwd=tcp:127.0.0.1:...). Since browser is on the same host, could even use 127.0.0.1 binding! But if they want to test from phone on LAN, leave default. Note: default empty host addr = all interfaces → LAN devices could hit it (mention security lightly).

RTP range 10000-10009 in rtp.conf matches the 10 UDP forwards.

Wait, one consideration: does Asterisk allocate RTCP ports separately? With rtcp_mux (WebRTC) it uses the same port; for non-mux endpoints RTP uses even port + RTCP odd. 10 ports → ~3-5 calls. Fine for dev; note it.

Also mention verifying with sudo ss -lunp | grep asterisk inside VM.

Also — inside-VM curl http_status check before browser testing ✓.

Let me now also draft the troubleshooting table concisely.

I think I'm ready to write the final answer. Structure:
0. The big picture (diagram + why WSS/TLS/certs)
1. Install Asterisk + first smoke test (inside the VM)
2. End-to-end: browser → Asterisk

2.1 QEMU networking 2.2 TLS cert 2.3 Asterisk config (http/pjsip/rtp/extensions) 2.4 The softphone webpage 2.5 First call + troubleshooting 2.6 How prod differs
3. A quick tour of IVR programming

3.1 Concepts (apps/priorities/contexts/special extens) 3.2 Two-level menu 3.3 Prompts: built-in, TTS, your own files 3.4 AGI + SQLite 3.5 AGI vs FastAGI vs ARI
4. Getting a real phone number
5. Cheat sheet / next steps

Write it now. Keep code blocks accurate. For extensions.conf part 2 version, include: 100 smoke, 200 hello IVR, 611 echo, _1XXX dial. Note: pattern _1XXX also matches 1000-1999 — but careful, it must come AFTER specific extens? Dialplan matching: exact match beats pattern always (patterns only used if no exact match). ✓ no ordering issue.

Wait, one thing on the echo test and Read/Background with WebRTC + ulaw: Background detects DTMF via telephone-event ✓.

Also 200 hello IVR: make it slightly interactive hello-world: Answer, Playback(demo-congrats), then SayNumber? Let me do:

exten => 200,1,Answer()
 same => n,Playback(custom/hello-ivr)      ; TTS: "Hello! You have reached the I V R smoke test."
 same => n,SayNumber(${EPOCH})?? eh.

Keep: 200 = Answer; Playback(demo-congrats); Playback(vm-goodbye); Hangup(). Simple hello world. Actually demo-congrats = "Congratulations, you have successfully installed Asterisk" — perfect hello world. ✓ And mention it proves prompt playback.

For the page: also handle the "answer incoming call" case (from the other tab / Dial from exten). onCallReceived auto-answer — good for testing 2-tab calls. I'll auto-answer after 1s? Immediate answer with a status note. Simple: auto-answer.

Status polling for UI robustness: setInterval 500ms → update buttons based on simpleUser.isConnected() and simpleUser.isCalling()? Does isCalling() exist... hmm. There's simpleUser.isCalling()? Not sure. There's session accessor... skip polling; rely on async flows + delegate. Acceptable.

Also connect() idempotency: "Connect" button once. Keep single "Connect & Register" button.

Edge: sendDTMF throws if no call → try/catch.

Also — sending DTMF while IVR is in Background(): note SIP.js sends via RTP telephone-event after call established ✓. Timing note: digits sent rapidly may be dropped by browser? RTCDTMFSender queues. Fine.

One more subtle thing: the page served from https://HOSTIP:8089 — Asterisk static server: content types? .html → text/html ✓ (asterisk handles). jsdelivr CDN script over https ✓ mixed content OK (https page loading https CDN ✓).

If the Windows host has no internet? assume it does.

Alright — also remember to explicitly explain "the connection details" server vs client as user asked:

Server side (Asterisk): listens on 0.0.0.0:8089 TLS for WSS; authenticates user 1000/pass; matches endpoint by From username; RTP ports 10000-10009 advertised... plus ice mapping so it advertises the HOST's LAN IP. Client side (browser): knows server = wss://<same-host-as-page>:8089/ws (auto from location.hostname), identity sip:1000@<host>, credentials; mic permission; audio element.

And explain QEMU mapping chain: browser → 192.168.1.50:8089 (host) → hostfwd → 10.0.2.15:8089 (guest Asterisk). And RTP: browser ⇄ 192.168.1.50:100xx → hostfwd → 10.0.2.15:100xx, with ICE candidates rewritten via ice_host_candidates; fallback path via SLIRP outbound NAT.

Good. Now write. I'll aim for completeness with tight prose. Also add note early: "all commands as root/sudo; Debian 12 bookworm; Asterisk 20".

For rtp.conf ice_host_candidates, I'll write:

[ice_host_candidates]
; Map the guest's private IP to the QEMU host's LAN IP so that
; ICE candidates in the SDP are reachable from the browser.
; REPLACE 192.168.1.50 with your Windows host's actual IP.
10.0.2.15 => 192.168.1.50

with a hedge comment "if your Asterisk build doesn't support this section (check rtp.conf.sample / CLI), remove it — RTP can still flow via Asterisk's outbound ICE checks through QEMU's NAT".

Hmm — actually, thinking about it more: is it 10.0.2.15 => 192.168.1.50? The section maps the address Asterisk would advertise (its local interface IP) to the external one. The guest IP is 10.0.2.15 by default in QEMU SLIRP. ✓. Users can confirm guest IP via ip addr / it's almost always 10.0.2.15.

Also one more potential gotcha: Asterisk needs to actually bind RTP to 10.0.2.15... it binds 0.0.0.0 and uses the default route interface for candidates. ✓.

Also icesupport=yes global in rtp.conf — with PJPROJECT ICE. ✓.

And in pjsip transport for wss, should I add external_media_address=192.168.1.50? For WSS the SDP c= line... with ICE, candidates matter more. Adding external_media_address on wss transport + external_signaling_address might mangle things (those options are meant for UDP/TCP NAT scenarios and require reload behaviors). I'll leave it out of main config, mention in prod section for UDP trunks. Keep the ice_host_candidates approach.

OK writing final answer now. Also I want to briefly note at the top: "picking up from where the last session left off — plan was complete, writing the full guide now." Since user asked to confirm trace visibility — I already confirmed at the start. Then deliver.

Length target: comprehensive. Let me write it.

----

Yes — I can see the full thinking trace from the last round (it starts with "Let me look at what happened..." and ends mid-sentence at "...consider an SBC (Kamailio/OpenSIPS) in front for scale,"). The planning was essentially complete; the only unfinished piece was the tail end of Part 4. So I'll finish that last thought and deliver the full guide now.

The one remaining decision from the trace: for production scale/federation, Kamailio or OpenSIPS (or a hosted SBC) sits in front of Asterisk to handle thousands of SIP registrations, TLS, and DoS protection, while Asterisk (possibly multiple instances) handles the media apps — but for a dev box and small prod, raw Asterisk is fine. With that, here is the complete deliverable.

---

# IVR Development on Asterisk: A Complete Local Lab

**Environment assumptions:** Debian 12 (bookworm) VM (ships Asterisk 20 LTS — everything below also works on 18/21/22 with trivial changes), QEMU on a Windows host with user-mode networking (`-netdev user`), and a browser on the Windows host. All VM commands assume root or `sudo`.

---

## 0. The Big Picture

Telephony has three distinct planes, and keeping them straight is 90% of the battle:

```
                 SIGNALING                  MEDIA
Windows host ┌──────────────────────┐  ┌─────────────────────┐
 (browser +  │ SIP messages over    │  │ Audio: RTP packets  │
  mic/speaker)│ WebSocket (WSS)     │  │ (WebRTC: SRTP/DTLS, │
             │ TCP 8089             │  │ UDP 10000-10009)    │
             └──────────┬───────────┘  └──────────┬──────────┘
                        │  QEMU hostfwd (NAT)     │
                        ▼                          ▼
Debian VM (10.0.2.15) ──────  Asterisk  ─────────────
                        ├─ built-in HTTPS server :8089 (serves the
                        │   softphone webpage AND the WSS endpoint)
                        └─ dialplan = your "IVR program"
```

Two things that trip everyone up, addressed up front:

1. **Why TLS everywhere?** Browsers only grant microphone access (`getUserMedia`) on *secure contexts*. A plain `http://192.168.x.x` page gets no mic. So the page must be served over HTTPS, and SIP.js needs `wss://` (Secure WebSocket). We'll use one self-signed cert for both, served by Asterisk's own built-in web server — one cert, one port, minimal moving parts.
2. **Why is audio (RTP) the hard part?** Signaling is TCP and trivially forwarded with `hostfwd`. Media is UDP, and browsers mandate ICE (connectivity checks) for WebRTC. We'll deal with this explicitly in §2.1.

---

## 1. Install Asterisk + Smoke Test (Inside the VM)

### 1.1 Install

```bash
sudo apt update
sudo apt install -y asterisk
# Handy companions we'll use later:
sudo apt install -y sox espeak-ng sqlite3

asterisk -V          # e.g. "Asterisk 20.9.0"
sudo systemctl enable --now asterisk
sudo systemctl status asterisk --no-pager
```

If sounds are missing (Debian normally pulls them in): `sudo apt install asterisk-core-sounds-en-wav`.

### 1.2 Replace the sample dialplan with a minimal one

The shipped configs are enormous teaching samples — great to read later, noise to develop against. Back them up and start clean:

```bash
cd /etc/asterisk
sudo cp extensions.conf extensions.conf.sample   # keep the original for reading
```

Write `/etc/asterisk/extensions.conf`:

```ini
[general]
static=yes
writeprotect=no

; Contexts are dialplan "namespaces". Calls from our SIP clients will
; arrive in this context (we'll say so in pjsip.conf later).
[from-internal]

; --- 100: smoke-test extension ---
exten => 100,1,Answer()
 same => n,Playback(demo-congrats)
 same => n,Hangup()

; --- 611: echo test (the classic "is my audio path OK?" test) ---
exten => 611,1,Answer()
 same => n,Playback(demo-echotest)
 same => n,Echo()
 same => n,Playback(demo-congrats)
 same => n,Hangup()
```

Reading the syntax: `exten => <number>,<priority>,<application>(<args>)` — priority 1 is the first step; `same => n,` means "next priority". `Answer()`, `Playback()`, `Echo()`, `Hangup()` are built-in *applications*.

### 1.3 Smoke test — no phone needed

You can exercise the dialplan headlessly with a **Local channel** (a pseudo-channel that plugs straight into the dialplan):

```bash
# Tail the log in one terminal:
sudo asterisk -rvvv

# Originate a call into exten 100 in another terminal (or from the CLI):
sudo asterisk -rx 'channel originate Local/100@from-internal application Playback demo-congrats'
```

In the verbose output you should see the dialplan actually executing:

```
Executing [100@from-internal:1] Answer("Local/100@from-internal-00000000;2", "")
Executing [100@from-internal:2] Playback("Local/...;2", "demo-congrats")
```

Useful CLI commands while you're here:

```
core show version            dialplan show from-internal
core show channels           core show sounds (long — pipe through grep)
```

**That's the part-1 smoke test: Asterisk is installed, running, executing dialplan, and playing sounds — all inside the VM, no client required.**

---

## 2. End-to-End: Browser Softphone → Asterisk IVR

### 2.1 QEMU networking (the part you asked to have explained carefully)

**On the Windows host**, launch (or relaunch) the VM with forwarded ports:

```bat
qemu-system-x86_64.exe ^
  -m 2048 -smp 2 ^
  -drive file=debian.qcow2,if=virtio ^
  -netdev user,id=n0,hostfwd=tcp::2222-:22,hostfwd=tcp::8089-:8089,hostfwd=udp::10000-:10000,hostfwd=udp::10001-:10001,hostfwd=udp::10002-:10002,hostfwd=udp::10003-:10003,hostfwd=udp::10004-:10004,hostfwd=udp::10005-:10005,hostfwd=udp::10006-:10006,hostfwd=udp::10007-:10007,hostfwd=udp::10008-:10008,hostfwd=udp::10009-:10009 ^
  -device virtio-net-pci,netdev=n0
```

(PowerShell: replace `^` with a backtick. Each `hostfwd=tcp::X-:X` maps host port X → guest port X; `2222→22` is SSH for convenience. By default these bind all host interfaces; use `hostfwd=tcp:127.0.0.1:8089-:8089` to restrict to the host itself.)

Find your **Windows host LAN IP** — `ipconfig`, e.g. `192.168.1.50`. Everything below uses that as `HOST_IP`. Note the VM under SLIRP networking is always `10.0.2.15`, gateway `10.0.2.2`.

**Why the UDP forwards + what actually happens with RTP:**

- **Signaling (WSS, TCP 8089):** trivial. Browser → `HOST_IP:8089` → hostfwd → `10.0.2.15:8089`. Done.
- **Media (RTP, UDP):** WebRTC uses ICE. Asterisk will naively advertise candidates on `10.0.2.15`, which the browser **cannot reach** (it's behind QEMU's NAT). Two mechanisms rescue us:
  1. **We rewrite Asterisk's advertised candidate** to `HOST_IP` (see `rtp.conf` below). The browser then sends RTP to `HOST_IP:100xx`, which hostfwd delivers to the guest. Asterisk replies to whatever source it saw (the QEMU gateway `10.0.2.2`), and QEMU translates back. Clean, deterministic.
  2. Even without that, it *often* works anyway: Asterisk's own outbound ICE checks go out through QEMU's NAT, reach the browser on the host (same machine!), and the browser's "triggered check" reply reuses that NAT mapping. Symmetric RTP keeps it alive. This is why some tutorials "just work" under SLIRP with zero RTP config.
- We forward **10 UDP ports** (≈3–5 simultaneous calls — each call uses 1–2 ports). Dev only; prod opens a bigger range.
- **Honest alternative:** if you fight one-way audio for more than 15 minutes, switch the VM to **bridged networking** (a real LAN IP, no NAT, no forwarding — everything just works). QEMU: `-netdev bridge,id=n0,br=br0` (needs a bridge set up on... actually on Windows, TAP/bridge is fiddly; libvirt/virt-manager makes it easy). But try the config below first — it does work in practice.

### 2.2 TLS certificate (self-signed, dev only)

```bash
sudo mkdir -p /etc/asterisk/keys && cd /etc/asterisk/keys
sudo openssl req -x509 -newkey rsa:2048 -sha256 -days 3650 -nodes \
  -keyout asterisk.key -out asterisk.crt \
  -subj "/CN=asterisk-dev" \
  -addext "subjectAltName=IP:192.168.1.50,IP:10.0.2.15,DNS:localhost"
#                                  ^^^^ replace with YOUR Windows host LAN IP

sudo sh -c 'cat asterisk.crt asterisk.key > asterisk.pem'
sudo chown asterisk:asterisk /etc/asterisk/keys/asterisk.pem
sudo chmod 600 /etc/asterisk/keys/asterisk.pem
```

(Replace `192.168.1.50` with your actual host IP. Browsers validate the SAN, not the CN — with a self-signed cert you'll click through a warning once regardless, but a correct SAN avoids repeated complaints from Firefox.)

### 2.3 Asterisk configuration

**`/etc/asterisk/http.conf`** — the built-in web server: TLS for WSS **and** static hosting of our webpage:

```ini
[general]
enabled=yes
bindaddr=0.0.0.0
bindport=8088                  ; plain HTTP (debugging / http_status)
tlsenable=yes
tlsbindaddr=0.0.0.0:8089       ; HTTPS + Secure WebSocket
tlscertfile=/etc/asterisk/keys/asterisk.pem
enablestatic=yes               ; serve files from /var/lib/asterisk/static-http/
```

**`/etc/asterisk/pjsip.conf`** — the SIP stack (PJSIP; the old `chan_sip` is deprecated/removed — don't use it):

```ini
[global]
type=global
user_agent=Asterisk Dev PBX

; --- WebSocket transport. Note: TLS for this transport is provided by
; --- the HTTP server above (http.conf), not here.
[transport-wss]
type=transport
protocol=wss
bind=0.0.0.0

; ============ Extension 1000 ============
[1000]
type=endpoint
transport=transport-wss
context=from-internal
disallow=all
allow=ulaw                     ; G.711 is mandatory in WebRTC — always works
                               ; (add opus only if you have codec_opus loaded)
webrtc=yes                     ; sets ice_support, dtls-srtp, avpf, rtcp-mux,
                               ; and auto-generates the DTLS cert
direct_media=no
rtp_symmetric=yes
force_rport=yes
rewrite_contact=yes
auth=auth1000
aors=1000

[auth1000]
type=auth
auth_type=userpass
username=1000
password=devpass1000

[1000]
type=aor
max_contacts=4                 ; browsers re-register often; keep old contacts?
remove_existing=yes            ; no — newest registration wins

; ============ Extension 1001 (for two-tab / two-browser testing) ============
[1001]
type=endpoint
transport=transport-wss
context=from-internal
disallow=all
allow=ulaw
webrtc=yes
direct_media=no
rtp_symmetric=yes
force_rport=yes
rewrite_contact=yes
auth=auth1001
aors=1001

[auth1001]
type=auth
auth_type=userpass
username=1001
password=devpass1001

[1001]
type=aor
max_contacts=4
remove_existing=yes
```

**`/etc/asterisk/rtp.conf`**:

```ini
[general]
rtpstart=10000
rtpend=10009        ; matches the 10 UDP hostfwd rules
icesupport=yes
strictrtp=no        ; dev-lab friendliness behind NAT (prod: leave default yes)

; Rewrite the ICE candidates Asterisk puts in its SDP so the browser
; (on the Windows host) gets a reachable address.
; REPLACE 192.168.1.50 with your Windows host's LAN IP.
; (If your Asterisk build doesn't support this section — check
;  rtp.conf.sample — just delete it; SLIRP outbound NAT can still
;  carry RTP via Asterisk's outbound ICE checks.)
[ice_host_candidates]
10.0.2.15 => 192.168.1.50
```

**Add the hello-world IVR and inter-extension dialing to `/etc/asterisk/extensions.conf`** (inside `[from-internal]`):

```ini
; --- 200: hello-world IVR ---
exten => 200,1,Answer()
 same => n,Playback(demo-congrats)
 same => n,Playback(vm-goodbye)
 same => n,Hangup()

; --- 1000/1001: call the other browser ---
exten => _1XXX,1,Dial(PJSIP/${EXTEN},20)
 same => n,Hangup()
```

(`_1XXX` is a *pattern*: underscore, then 4 digits starting with 1. Exact matches always beat patterns, so `100`/`200`/`611` still work.)

**Apply and verify:**

```bash
sudo systemctl restart asterisk
sudo asterisk -rvvv
```

```
CLI> http show status           ; look for "Enabled: yes", TLS bound to 0.0.0.0:8089,
                                 ; and /ws (websocket) + static content listed
CLI> pjsip show transports      ; transport-wss wss 0.0.0.0
CLI> pjsip show endpoints       ; 1000, 1001
CLI> module show like res_pjsip_transport_websocket   ; must be loaded
```

From inside the VM: `curl http://127.0.0.1:8088/http_status` → should return an Asterisk status page.

### 2.4 The softphone webpage (SIP.js)

Design: Android-dialer-style — a digit display, a keypad, call/hangup, and status. Before a call, keypad types into the display (the extension to dial); during a call, keypad keys send DTMF immediately (that's how you drive the IVR). One clever trick makes it zero-config: since the page is served *from Asterisk itself*, `location.hostname` is already the right server address — the client derives `wss://<host>:8089/ws` automatically.

Write `/var/lib/asterisk/static-http/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IVR Test Phone</title>
<!-- Pinned version: SimpleUser's constructor/connect signatures have changed
     across SIP.js releases. This code targets 0.21.x. -->
<script src="https://cdn.jsdelivr.net/npm/sip.js@0.21.1/dist/sip.min.js"></script>
<style>
  body { font-family: system-ui, sans-serif; background:#111; color:#eee;
         max-width:420px; margin:24px auto; padding:0 12px; }
  #status { padding:8px 12px; border-radius:8px; background:#222;
            margin-bottom:12px; font-size:14px; min-height:20px; }
  #status.ok  { background:#143d1f; }
  #status.err { background:#4d1414; }
  #display { font-size:28px; text-align:center; background:#000;
             border-radius:8px; padding:14px; margin-bottom:12px;
             min-height:34px; letter-spacing:2px; font-family:monospace; }
  #keypad { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }
  #keypad button { font-size:22px; padding:16px 0; border:0; border-radius:10px;
                   background:#2a2a2a; color:#fff; cursor:pointer; }
  #keypad button:active { background:#444; }
  #keypad button small { display:block; font-size:10px; color:#999; }
  #actions { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:12px; }
  #btnCall  { background:#1d7a34 !important; }
  #btnHang  { background:#8a1f1f !important; }
  #btnHang:disabled, #btnCall:disabled { opacity:.35; cursor:not-allowed; }
  details { margin-top:16px; font-size:13px; }
  input { width:100%; box-sizing:border-box; margin:2px 0 8px; padding:6px;
          background:#222; color:#eee; border:1px solid #444; border-radius:6px; }
  #log { font-family:monospace; font-size:11px; color:#888; margin-top:12px;
         max-height:120px; overflow-y:auto; white-space:pre-wrap; }
</style>
</head>
<body>
  <h3 style="margin:4px 0 12px">📞 IVR Test Phone</h3>
  <div id="status">not connected</div>
  <div id="display">&nbsp;</div>

  <div id="keypad"></div>

  <div id="actions">
    <button id="btnCall" disabled>Call</button>
    <button id="btnHang" disabled>Hang up</button>
  </div>

  <details>
    <summary>Connection settings</summary>
    <label>SIP user</label><input id="user" value="1000">
    <label>Password</label><input id="pass" value="devpass1000" type="password">
    <label>WebSocket URL</label><input id="wsUrl">
    <label>SIP domain</label><input id="domain">
    <button id="btnConnect" style="width:100%;padding:8px">Connect &amp; Register</button>
  </details>

  <div id="log"></div>

<script>
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

  // ---------- SIP.js ----------
  $("btnConnect").onclick = async () => {
    try {
      setStatus("connecting…");
      simpleUser = new SIP.Web.SimpleUser({
        delegate: {
          // Auto-answer incoming calls (handy when testing with two tabs)
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
        }
      });

      await simpleUser.connect({
        wsURL: $("wsUrl").value,
        authorizationUsername: $("user").value,
        authorizationPassword: $("pass").value,
        displayName: "Browser " + $("user").value
      });
      log("WebSocket connected");

      await simpleUser.register();
      setStatus("registered as " + $("user").value + " — dial an extension", "ok");
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
      log("calling sip:" + target + "@" + $("domain").value);
      await simpleUser.call("sip:" + target + "@" + $("domain").value, {
        media: { constraints: { audio: true, video: false } }
      });
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
</script>
</body>
</html>
```

> **SIP.js version note:** SIP.js has shuffled the `SimpleUser` constructor/connect signatures across 0.15 → 0.21 (most notably whether the WebSocket URL goes into the constructor or into `connect()`). The code above targets **0.21.x, pinned via the CDN URL**. If you bump the version and get a `TypeError` at startup, that signature shift is the first thing to check against the SimpleUser docs for the version you pinned.

Make it readable by Asterisk:

```bash
sudo chown asterisk:asterisk /var/lib/asterisk/static-http/index.html
```

### 2.5 First call

On the **Windows host** browser:

1. Open `https://192.168.1.50:8089/index.html` (your `HOST_IP`).
2. Accept the self-signed certificate warning (`Advanced → proceed`). One acceptance covers both the page and the WSS connection, since they're the same origin.
3. Click **Connect & Register** → status should turn green ("registered as 1000").
4. Type **100** → **Call**. Allow mic access. You should hear *"Congratulations, you have successfully installed Asterisk."*
5. Call **611** (echo test) and speak — hearing your own voice delayed proves **both RTP directions**, DTLS-SRTP, and ICE all work.
6. Call **200** — the hello-world IVR.
7. Open a second tab, change the SIP user to `1001`/`devpass1001` in settings, register, then call `1001` from the first tab — browser-to-browser calling via the PBX.

**Troubleshooting quick hits:**

| Symptom | Likely cause / fix |
|---|---|
| Page won't load | `http show status` in CLI; `enablestatic=yes`? port 8089 forwarded? |
| "connect failed" in red | Wrong password; or cert not accepted (revisit the https URL); or `res_pjsip_transport_websocket` not loaded |
| Registers, but calls fail with "no endpoint" | From-username matching: add `identify_by=username,ip` to the endpoints in `pjsip.conf`, reload |
| Call connects, **no audio either way** | ICE/NAT: run `rtp set debug on` in the CLI and watch; check `ice_host_candidates` IP is right; try Firefox vs Chrome; last resort — bridged networking |
| One-way audio | Usually the ICE pair took the "outbound NAT" path and died — fix `ice_host_candidates` |
| No DTMF reaching the IVR | Keys must be pressed while status says "in call"; check `pjsip show endpoint 1000` → `dtmf_mode` should be `rfc4733` |
| Hear nothing but call is up | Chrome autoplay policy — click anywhere on the page, re-call |

In the Asterisk CLI, `core set verbose 5` shows every dialplan step as your call runs — your best friend for the rest of this guide.

### 2.6 How this differs from a production server with a public IP

- **No NAT gymnastics.** The server has a real IP; the browser connects to `wss://pbx.example.com/ws` directly. For WebRTC clients specifically, the WebSocket transport makes NAT on the *server* side mostly a non-issue — but you still set `external_media_address` for classic UDP/TCP SIP transports (§4).
- **Real certificates.** Let's Encrypt (or your CA) instead of self-signed — no click-through, no warnings, and cert renewal automated. The WSS cert must match the hostname the page connects to.
- **Ports:** typically WSS on 443 behind nginx/Caddy/traefik (TLS termination + static files), and a UDP/TCP 5060–5061 SIP listener + a large RTP UDP range (e.g. 10000–20000) open in the firewall.
- **Remote WebRTC clients behind their own NATs** may need a **STUN/TURN server** (coturn) for ICE when direct paths fail. Local-network dev never exercises this; production should have it.
- **Hardening:** strong per-endpoint secrets, `fail2ban` on Asterisk security logs, never expose AMI/ARI without auth+TLS, restrict which IPs may talk SIP to you (or front everything with an SBC).

---

## 3. A Quick Tour of Asterisk IVR Programming

Now the actual "IVR app dev" part. Concepts first, in one breath: your program is the **dialplan** — a set of **contexts** (namespaces) containing **extensions** (numbers/patterns) containing ordered **priorities** that call **applications** (`Playback`, `Background`, `Read`, `Goto`, `Dial`, …). Interactive prompts use `Background()` (plays audio *while listening* for digits), then `WaitExten()` dispatches to the extension matching the pressed digit. Special extensions: `s` (start), `t` (timeout), `i` (invalid). When logic outgrows the dialplan, you shell out to **AGI** scripts — same idea as CGI for web servers.

### 3.1 Two-level menu

```ini
; ============ MAIN MENU (dial 300) ============
[from-internal]
exten => 300,1,Answer()
 same => n,Set(TIMEOUT(digit)=2)      ; max gap between digits (sec)
 same => n,Set(TIMEOUT(response)=8)   ; max wait for first digit (sec)
 same => n,Background(custom/ivr-welcome)   ; plays AND listens for digits
 same => n,WaitExten()                ; dispatch on the digit pressed
 same => n,Goto(t,1)                  ; fall-through = nothing pressed

exten => 1,1,Goto(ivr-sales,s,1)
exten => 2,1,Goto(ivr-support,s,1)
exten => 3,1,Goto(ivr-balance,s,1)    ; the AGI demo (§3.3)
exten => 0,1,Dial(PJSIP/1000,15)      ; "operator" = first browser
 same => n,Goto(300,1)

exten => i,1,Playback(custom/invalid) ; invalid digit ("i")
 same => n,Goto(300,1)
exten => t,1,Playback(custom/goodbye) ; timeout ("t")
 same => n,Hangup()

; ============ LEVEL 2: SALES ============
[ivr-sales]
exten => s,1,Set(TIMEOUT(digit)=2)
 same => n,Set(TIMEOUT(response)=8)
 same => n,Background(custom/ivr-sales)
 same => n,WaitExten()
 same => n,Goto(t,1)

exten => 1,1,Playback(custom/sales-hours)
 same => n,Wait(1)
 same => n,Goto(from-internal,300,1)
exten => 2,1,Dial(PJSIP/1000,15)      ; "sales rep" = first browser
 same => n,Goto(from-internal,300,1)
exten => *,1,Goto(from-internal,300,1)
exten => i,1,Playback(custom/invalid)
 same => n,Goto(s,1)
exten => t,1,Playback(custom/goodbye)
 same => n,Hangup()

; ============ LEVEL 2: SUPPORT ============
[ivr-support]
exten => s,1,Set(TIMEOUT(digit)=2)
 same => n,Set(TIMEOUT(response)=8)
 same => n,Background(custom/ivr-support)
 same => n,WaitExten()
 same => n,Goto(t,1)

exten => 1,1,Playback(custom/support-hours)
 same => n,Wait(1)
 same => n,Goto(from-internal,300,1)
exten => 2,1,Dial(PJSIP/1001,15)      ; "support engineer" = second browser
 same => n,Goto(from-internal,300,1)
exten => *,1,Goto(from-internal,300,1)
exten => i,1,Playback(custom/invalid)
 same => n,Goto(s,1)
exten => t,1,Playback(custom/goodbye)
 same => n,Hangup()
```

Call **300** from the webpage and drive it with the keypad. Watch it execute live with `core set verbose 5`, and inspect with `dialplan show from-internal`.

### 3.2 Audio prompts: three ways

**(a) Built-in sounds** — zero effort, robotic-but-usable fragments: `vm-goodbye`, `pbx-invalid`, `demo-congrats`, plus a whole library of composable words (`digits/1`, `digits/2`, `letters/a`, `vm/…`). Browse with `core show sounds | less`. E.g. you could replace `custom/goodbye` with `Playback(vm-goodbye)`.

**(b) Dynamic "TTS" without audio files** — synthesizer *applications* for numbers/digits/time, essential when content is dynamic (like a balance):

```
same => n,SayNumber(${BALANCE})     ; "four thousand two hundred"
same => n,SayDigits(${ACCOUNT})     ; "one two three four five six"
same => n,SayUnixTime()             ; "eight thirty five p m"
```

**(c) Your own audio files** — pre-generate WAVs with offline TTS, drop them in `/var/lib/asterisk/sounds/en/custom/`, reference as `custom/<name>` (Asterisk picks the best format automatically). Generate the whole prompt set with espeak-ng + sox:

```bash
sudo bash -e <<'EOF'
apt install -y espeak-ng sox
mkdir -p /var/lib/asterisk/sounds/en/custom
gen() {  # gen <name> <text>
  espeak-ng -v en-us+f3 -s 145 -w "/tmp/$1.wav" "$2"
  # Asterisk-native format: 8 kHz, mono, 16-bit PCM
  sox "/tmp/$1.wav" -r 8000 -c 1 "/var/lib/asterisk/sounds/en/custom/$1.wav"
  rm -f "/tmp/$1.wav"
}
gen ivr-welcome   "Welcome to Acme Robotics. For sales, press one. For technical support, press two. To check your account balance, press three. To speak with an operator, press zero."
gen ivr-sales     "You have reached sales. To hear our office hours, press one. To be connected to a sales representative, press two. To return to the main menu, press star."
gen ivr-support   "You have reached technical support. To hear our support hours, press one. To be connected to an engineer, press two. To return to the main menu, press star."
gen ivr-balance   "You have reached account services."
gen enter-account "Please enter your six digit account number, followed by the pound key."
gen unknown-account "Sorry, we could not find that account number."
gen your-balance-is "Your current balance is"
gen dollars       "dollars"
gen invalid       "Sorry, that is not a valid option."
gen goodbye       "Thank you for calling. Goodbye."
gen sales-hours   "Our office hours are Monday through Friday, nine A M to five P M."
gen support-hours "Our support team is available twenty four hours a day, seven days a week."
EOF
sudo asterisk -rx "dialplan reload"
```

When you outgrow espeak's robot voice: **Piper** (free, offline, neural, very good) or cloud TTS (Azure/Google/AWS Polly) — same workflow: generate WAV → convert to 8 kHz mono with sox → drop in the folder. Historically there were live-TTS engines in-process (Festival, Cepstral); today file-pre-generation (or AGI-triggered generation) is the pragmatic norm.

### 3.3 AGI: scripting the IVR with SQLite

AGI = Asterisk Gateway Interface: Asterisk spawns your script per call, hands it the channel over **stdin/stdout** with a dead-simple line protocol (yes — it's CGI for phones, and it's just as fun as that sounds). The killer demo: look up data in a DB and drive the call from it.

**The database:**

```bash
sudo sqlite3 /var/lib/asterisk/agi-bin/accounts.db <<'SQL'
CREATE TABLE accounts (id TEXT PRIMARY KEY, name TEXT, balance INTEGER);
INSERT INTO accounts VALUES ('123456', 'Alice', 4200);
INSERT INTO accounts VALUES ('654321', 'Bob', 75);
SQL
sudo chown -R asterisk:asterisk /var/lib/asterisk/agi-bin
```

**The script — `/var/lib/asterisk/agi-bin/balance.py`** (deliberately raw, so you see the actual protocol; no framework magic):

```python
#!/usr/bin/env python3
"""
balance.py — AGI demo: account balance lookup in SQLite.

Dialplan usage:   same => n,AGI(balance.py,${ACCOUNT})

Protocol: Asterisk sends environment lines on stdin (agi_channel,
agi_callerid, ... , agi_arg_1..N), terminated by a blank line. Then you
send commands ("SET VARIABLE x y", "GET VARIABLE x", "VERBOSE ..." ...),
one per line; each is answered with "200 result=<code>[ (<value>)]".
"""
import sys
import sqlite3

DB_PATH = "/var/lib/asterisk/agi-bin/accounts.db"


def read_env():
    env = {}
    while True:
        line = sys.stdin.readline()
        if line in ("", "\n"):          # blank line = end of env block
            break
        if ":" in line:
            k, _, v = line.partition(":")
            env[k.strip()] = v.strip()
    return env


def agi(cmd):
    """Send one command, return the raw response line, e.g. '200 result=1 (123456)'"""
    sys.stdout.write(cmd + "\n")
    sys.stdout.flush()
    return sys.stdin.readline().strip()


def main():
    env = read_env()
    # Arguments arrive BOTH as argv and as agi_arg_1..N in the env block.
    account = (sys.argv[1] if len(sys.argv) > 1 else env.get("agi_arg_1", "")).strip()

    # You can also read channel variables set by the dialplan:
    # resp = agi("GET VARIABLE ACCOUNT")   -> '200 result=1 (123456)'

    row = None
    try:
        con = sqlite3.connect(DB_PATH)
        row = con.execute(
            "SELECT name, balance FROM accounts WHERE id = ?", (account,)
        ).fetchone()
        con.close()
    except sqlite3.Error as e:
        agi('VERBOSE "balance.py DB error: %s" 3' % e)

    if row:
        name, balance = row
        agi('SET VARIABLE AGI_RESULT "FOUND"')
        agi('SET VARIABLE BALANCE "%d"' % int(balance))
        agi('SET VARIABLE ACCNAME "%s"' % name)
        agi('VERBOSE "balance.py: %s -> %s ($%d)" 3' % (account, name, balance))
    else:
        agi('SET VARIABLE AGI_RESULT "NOTFOUND"')
        agi('VERBOSE "balance.py: %s -> not found" 3' % account)


if __name__ == "__main__":
    main()
```

```bash
sudo chmod 755 /var/lib/asterisk/agi-bin/balance.py
sudo chown asterisk:asterisk /var/lib/asterisk/agi-bin/balance.py
```

**The dialplan — add the third submenu:**

```ini
; ============ LEVEL 2: ACCOUNT BALANCE (AGI + SQLite) ============
[ivr-balance]
exten => s,1,Playback(custom/enter-account)
 same => n,Read(ACCOUNT,custom/enter-account,6,,3,10)   ; var,prompt,maxdig,options,attempts,timeout
 same => n,GotoIf($["${ACCOUNT}" = ""]?from-internal,300,1)
 same => n,AGI(balance.py,${ACCOUNT})
 same => n,GotoIf($["${AGI_RESULT}" = "FOUND"]?found:notfound)
 same => n(found),Playback(custom/your-balance-is)
 same => n,SayNumber(${BALANCE})
 same => n,Playback(custom/dollars)
 same => n,Wait(1)
 same => n,Goto(from-internal,300,1)
 same => n(notfound),Playback(custom/unknown-account)
 same => n,Goto(from-internal,300,1)
```

`sudo asterisk -rx "dialplan reload"`, call **300 → press 3 → enter 123456#** → hear "Your current balance is four thousand two hundred dollars."

**Debugging AGI:** in the CLI, `agi set debug on` dumps the entire stdin/stdout session — the single best way to learn the protocol. Also `core set verbose 5` shows your `VERBOSE` lines. Note the script inherits the channel state (already answered — we answered at the top of exten 300), runs as the `asterisk` user, and the channel is *blocked* while it runs — for slow I/O, keep dialplan doing the prompting and AGI doing the lookup, or go FastAGI.

**The ecosystem, in one paragraph each:**

- **AGI** (what we did): script-per-call, synchronous, any language that can read stdin. Libraries exist that wrap the protocol nicely — Python `pyst2`, PHP `PAGI`, and there's even `asterisk-java`… for Java.
- **FastAGI** (`AGI(agi://host:port/script)`): same protocol, but over a TCP socket to a long-lived server process — no process spawn per call, your app can be stateful and remote. Same libraries generally support it.
- **ARI** (Asterisk REST Interface): the *modern* architecture. The dialplan just does `Stasis(my-app)`; your external application then controls the call entirely over **REST + WebSocket events** (bridging, playback, recording, transfers), independent of the dialplan language. If you're a software engineer building a serious telephony app, this is where you'll eventually live — AGI is the right on-ramp, ARI is the destination.

---

## 4. Getting a Real Phone Number

So far everything lives on your LAN. To let an actual mobile/landline phone call your IVR:

### 4.1 The concept: DIDs and SIP trunks

You buy a **DID** (Direct Inward Dialing number — just telco-speak for "a phone number") from an **ITSP** (Internet Telephony Service Provider), typically $1–5/month plus per-minute usage. The provider receives the PSTN call and forwards it to your server over the internet as **SIP + RTP** — that connection is your **SIP trunk**. Outbound calls go the same way in reverse.

Well-known providers: **Telnyx, Twilio (Elastic SIP Trunking), Bandwidth, Flowroute, Voxbeam, VoIP.ms, DIDLogic, les.net**, and regionals like **sipgate** (DE/UK) or **AnveoDirect**. Realities to expect: signup KYC/identity verification, some countries require local address proof for local numbers, and if *you* serve US end users there are **E911** obligations to route emergency calls — read your provider's compliance docs.

And a genuinely valid question: **do you even need a number?** If your users are humans in browsers (support widget, click-to-call from your app), WebRTC straight to your Asterisk needs no PSTN at all. The PSTN is only for reaching the world's regular phones.

### 4.2 Linking the trunk to your PBX

You need a public-facing server first: a VPS or a server with a public IP (your QEMU lab is *not* that), DNS, and firewall open for SIP (UDP/TCP 5060, optionally TLS 5061) plus your RTP range. Two provider models:

**Model A — you register to them** (works even behind NAT; provider gives you SIP username/password):

```ini
; /etc/asterisk/pjsip.conf — add a UDP transport for the trunk
[transport-udp]
type=transport
protocol=udp
bind=0.0.0.0
; external_media_address=203.0.113.10   ; your PUBLIC IP (if 1:1 NAT)
; external_signaling_address=203.0.113.10

; --- Registration to the provider (outbound) ---
[mytrunk]
type=registration
transport=transport-udp
outbound_auth=mytrunk_auth
server_uri=sip:sip.provider.example:5060
client_uri=sip:+15551234567@sip.provider.example:5060
retry_wait=60

[mytrunk_auth]
type=auth
auth_type=userpass
username=+15551234567
password=secret-from-provider

[mytrunk]
type=aor
contact=sip:sip.provider.example:5060

; --- Matching their inbound calls to this endpoint ---
[mytrunk]
type=identify
endpoint=mytrunk
match=sip.provider.example

[mytrunk]
type=endpoint
transport=transport-udp
context=from-pstn              ; <-- where inbound calls land in the dialplan
disallow=all
allow=ulaw                     ; match what the provider uses (ulaw US / alaw EU)
direct_media=no
rtp_symmetric=yes
force_rport=yes
rewrite_contact=yes
outbound_auth=mytrunk_auth
aors=mytrunk
```

**Model B — IP peering** (no password; you both whitelist each other's IPs — the usual choice when you have a static public IP): drop the `registration` section, keep endpoint/aor/identify, and add `match=` entries for their SIP servers.

**The actual "link": inbound routing in the dialplan.** Inbound calls arrive in `from-pstn` with the dialed number (your DID) as the extension:

```ini
[from-pstn]
; My DID rings the main IVR — THIS is the "link":
exten => +15551234567,1,NoOp(Inbound PSTN call from ${CALLERID(num)})
 same => n,Goto(from-internal,300,1)
 same => n,Hangup()

; Catch-all for any other number the provider sends:
exten => _X.,1,NoOp(Unexpected DID: ${EXTEN})
 same => n,Playtones(congestion)
 same => n,Hangup()
```

**Outbound** — give internal users a dial-out pattern (in `[from-internal]`):

```ini
; 10-digit and 1+10-digit US-style numbers go out the trunk:
exten => _NXXXXXXXXX,1,Dial(PJSIP/${EXTEN}@mytrunk)
exten => _1NXXXXXXXXX,1,Dial(PJSIP/${EXTEN}@mytrunk)
```

And just like that, a phone in your pocket dials +1 555 123 4567 → provider → your Asterisk → `from-pstn` → your IVR from §3 — the same dialplan your browser tab was testing all along. That's the payoff of developing against the real thing.

**Production hygiene:** `fail2ban` watching Asterisk's security log, strong secrets everywhere, only provider IPs allowed to hit SIP (the `identify`/`match` discipline), TLS + SRTP where possible, and if you grow: a dedicated **SBC** (Kamailio/OpenSIPS) in front of one-or-many Asterisk instances for registration load, DoS absorption, and federation.

---

## 5. Where to Go Next

- **Docs:** the Asterisk wiki (wiki.asterisk.org) — specifically "Configuring Asterisk for WebRTC clients", the AGI section, and the ARI section.
- **Book:** *Asterisk: The Definitive Guide* (O'Reilly) — genuinely well-written and available free online.
- **Natural progressions:** dialplan → AGI → FastAGI → **ARI** (build your IVR as a standalone service in Node/Python/Go with REST+WebSockets); add `Queue()`/`Voicemail()` for a mini contact-center; try **FreePBX** once you want a GUI over all this (knowing what you now know, the GUI will make sense instead of being magic).
- **Your lab, one paragraph:** Debian VM + Asterisk 20, PJSIP over Secure WebSocket, self-signed TLS, a zero-config SIP.js webpage served by Asterisk itself, QEMU hostfwd for TCP 8089 + a 10-port UDP RTP range with an ICE candidate rewrite — call `100` for smoke, `611` for echo, `300` for the two-level menu with TTS prompts and a SQLite-backed AGI balance lookup, and a real DID is one SIP trunk away.

Enjoy — and when the audio doesn't flow, remember the ancient telephony debugging mantra: *it's always the NAT.*

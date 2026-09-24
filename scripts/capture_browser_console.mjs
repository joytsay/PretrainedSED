#!/usr/bin/env node

// Persist Chromium DevTools Protocol events outside the browser process.
// Requires Node.js 22+ and Chromium launched with a local debugging port.
import { appendFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

function option(name, fallback) {
  const index = process.argv.indexOf(name);
  if (index < 0) return fallback;
  if (!process.argv[index + 1]) throw new Error(`${name} needs a value`);
  return process.argv[index + 1];
}

const targetUrl = option('--url', '');
const outputPath = resolve(option('--output', 'browser-console.jsonl'));
const port = Number(option('--port', '9222'));
if (!targetUrl || !Number.isInteger(port) || port < 1 || port > 65535) {
  console.error('Usage: node scripts/capture_browser_console.mjs --url http://AGX-IP:8080/?debug=1 [--output browser-console.jsonl] [--port 9222]');
  process.exit(2);
}
if (typeof WebSocket === 'undefined') {
  console.error('Node.js 22 or newer is required for its built-in WebSocket client.');
  process.exit(2);
}

mkdirSync(dirname(outputPath), { recursive: true });
function record(kind, data = {}) {
  const line = JSON.stringify({ time: new Date().toISOString(), kind, ...data });
  appendFileSync(outputPath, `${line}\n`);
  if (kind === 'console' || kind === 'exception' || kind === 'network-failed' || kind === 'target-crashed') {
    console.log(line);
  }
}

let socket = null;
let attachedTarget = '';
let connecting = false;
let commandId = 0;
const requests = new Map();

function send(method) {
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ id: ++commandId, method }));
  }
}

function remoteValue(value) {
  if (Object.hasOwn(value, 'value')) return value.value;
  return value.unserializableValue ?? value.description ?? value.preview ?? value.type;
}

function handleMessage(raw) {
  let message;
  try { message = JSON.parse(raw); }
  catch { return; }
  const params = message.params ?? {};
  switch (message.method) {
    case 'Runtime.consoleAPICalled':
      record('console', {
        level: params.type,
        text: (params.args ?? []).map(remoteValue).join(' '),
        arguments: (params.args ?? []).map(remoteValue),
        source: params.stackTrace?.callFrames?.[0]
      });
      break;
    case 'Runtime.exceptionThrown':
      record('exception', {
        text: params.exceptionDetails?.text,
        exception: remoteValue(params.exceptionDetails?.exception ?? {}),
        source: params.exceptionDetails?.stackTrace?.callFrames?.[0]
      });
      break;
    case 'Log.entryAdded':
      record('browser-log', { entry: params.entry });
      break;
    case 'Network.requestWillBeSent':
      requests.set(params.requestId, { url: params.request?.url, method: params.request?.method });
      if (requests.size > 5000) requests.delete(requests.keys().next().value);
      break;
    case 'Network.responseReceived':
      if (params.response?.status >= 400) {
        record('http-error', {
          request: requests.get(params.requestId),
          status: params.response.status,
          statusText: params.response.statusText
        });
      }
      break;
    case 'Network.loadingFailed':
      record('network-failed', {
        request: requests.get(params.requestId),
        error: params.errorText,
        canceled: params.canceled
      });
      requests.delete(params.requestId);
      break;
    case 'Network.loadingFinished':
      requests.delete(params.requestId);
      break;
    case 'Inspector.targetCrashed':
      record('target-crashed', { status: params.status, errorCode: params.errorCode });
      break;
  }
}

async function discover() {
  if (socket || connecting) return;
  connecting = true;
  try {
    const response = await fetch(`http://127.0.0.1:${port}/json/list`, { signal: AbortSignal.timeout(2000) });
    if (!response.ok) return;
    const targets = await response.json();
    const target = targets.find((item) => item.type === 'page' && item.url?.startsWith(targetUrl) && item.webSocketDebuggerUrl);
    if (!target) return;
    const connection = new WebSocket(target.webSocketDebuggerUrl);
    socket = connection;
    attachedTarget = target.id;
    connection.addEventListener('open', () => {
      record('attached', { url: target.url, targetId: target.id });
      for (const method of ['Runtime.enable', 'Log.enable', 'Network.enable', 'Inspector.enable']) send(method);
    });
    connection.addEventListener('message', (event) => handleMessage(event.data));
    connection.addEventListener('close', () => {
      record('detached', { targetId: attachedTarget });
      socket = null;
      attachedTarget = '';
      requests.clear();
    });
    connection.addEventListener('error', (error) => {
      record('connection-error', { message: error.message ?? 'WebSocket error' });
    });
  } catch {
    // Chromium may be starting, stopped, or crashing. Retry quietly.
  } finally {
    connecting = false;
  }
}

record('recorder-started', { url: targetUrl, outputPath, port });
console.log(`Waiting for ${targetUrl}; writing ${outputPath}. Press Ctrl+C to stop.`);
void discover();
setInterval(() => void discover(), 1000);

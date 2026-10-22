import assert from 'node:assert/strict';
import test from 'node:test';
import { TriggerTiming, type TriggerClock } from '../src/components/trigger-timing.ts';

function fixture() {
  let now = 0;
  let id = 0;
  const tasks = new Map<number, { at: number; callback: () => void }>();
  const clock: TriggerClock = {
    now: () => now,
    schedule(delay, callback) {
      const key = ++id;
      tasks.set(key, { at: now + delay, callback });
      return () => { tasks.delete(key); };
    }
  };
  const events: Array<{ name: string; at: number }> = [];
  const gate = new TriggerTiming(name => events.push({ name, at: now }), clock);
  return {
    gate, events,
    advance(ms: number) {
      const target = now + ms;
      while (true) {
        const next = [...tasks].filter(([, task]) => task.at <= target).sort((a, b) => a[1].at - b[1].at)[0];
        if (!next) break;
        tasks.delete(next[0]);
        now = next[1].at;
        next[1].callback();
      }
      now = target;
    }
  };
}

test('continuous score updates do not restart the on delay', () => {
  const { gate, events, advance } = fixture();
  for (let i = 0; i < 5; i += 1) { gate.update(['Alarm'], 200, 200); advance(40); }
  assert.deepEqual(events, [{ name: 'Alarm', at: 200 }]);
  gate.update(['Alarm'], 200, 200); advance(1000);
  assert.equal(events.length, 1);
});

test('a confidence spike shorter than the on delay is discarded', () => {
  const { gate, events, advance } = fixture();
  gate.update(['Alarm'], 200, 200); advance(100);
  gate.update([], 200, 200); advance(500);
  assert.deepEqual(events, []);
  gate.update(['Alarm'], 200, 200); advance(199);
  assert.equal(events.length, 0); advance(1);
  assert.deepEqual(events, [{ name: 'Alarm', at: 800 }]);
});

test('a brief confidence dip does not rearm an already announced class', () => {
  const { gate, events, advance } = fixture();
  gate.update(['Alarm'], 200, 200); advance(200);
  gate.update([], 200, 200); advance(100);
  gate.update(['Alarm'], 200, 200); advance(1000);
  assert.equal(events.length, 1);
});

test('the full off delay rearms the class, followed by a fresh full on delay', () => {
  const { gate, events, advance } = fixture();
  gate.update(['Alarm'], 200, 200); advance(200);
  gate.update([], 200, 200); advance(200);
  gate.update(['Alarm'], 200, 200); advance(199);
  assert.equal(events.length, 1); advance(1);
  assert.deepEqual(events, [{ name: 'Alarm', at: 200 }, { name: 'Alarm', at: 600 }]);
});

test('each class keeps independent on and off timers', () => {
  const { gate, events, advance } = fixture();
  gate.update(['Alarm'], 200, 200); advance(100);
  gate.update(['Alarm', 'Glass'], 200, 200); advance(100);
  gate.update(['Glass'], 200, 200); advance(100);
  assert.deepEqual(events, [{ name: 'Alarm', at: 200 }, { name: 'Glass', at: 300 }]);
  advance(100); gate.update(['Alarm', 'Glass'], 200, 200); advance(200);
  assert.deepEqual(events.at(-1), { name: 'Alarm', at: 600 });
  assert.equal(events.filter(event => event.name === 'Glass').length, 1);
});

test('editing delays preserves the time already spent above or below threshold', () => {
  const { gate, events, advance } = fixture();
  gate.update(['Alarm'], 200, 200); advance(50);
  gate.update(['Alarm'], 100, 200); advance(50);
  assert.deepEqual(events, [{ name: 'Alarm', at: 100 }]);
  gate.update([], 100, 200); advance(100);
  gate.update([], 100, 400); advance(299);
  gate.update(['Alarm'], 100, 400); advance(200);
  assert.equal(events.length, 1, 'The longer off delay was not reached');
});

test('reset cancels pending triggers and clears latched classes on media end or disable', () => {
  const { gate, events, advance } = fixture();
  gate.update(['Alarm'], 200, 200); advance(100); gate.reset(); advance(1000);
  assert.deepEqual(events, []);
  gate.update(['Alarm'], 200, 200); advance(200);
  gate.reset(); gate.update(['Alarm'], 200, 200); advance(200);
  assert.equal(events.length, 2);
});

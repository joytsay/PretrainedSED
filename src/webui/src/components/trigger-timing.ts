type CancelTimer = () => void;
export interface TriggerClock {
  now(): number;
  schedule(delayMs: number, callback: () => void): CancelTimer;
}
const browserClock: TriggerClock = {
  now: () => performance.now(),
  schedule(delayMs, callback) {
    const timer = setTimeout(callback, delayMs);
    return () => clearTimeout(timer);
  }
};
type ClassState = {
  above: boolean;
  latched: boolean;
  since: number;
  deadline: number | null;
  cancel: CancelTimer | null;
};

/** Independent sustained-on and sustained-off timers for each detector class. */
export class TriggerTiming {
  private states = new Map<string, ClassState>();
  private emit: (name: string) => void;
  private clock: TriggerClock;

  constructor(emit: (name: string) => void, clock: TriggerClock = browserClock) {
    this.emit = emit;
    this.clock = clock;
  }

  reset(): void {
    for (const state of this.states.values()) state.cancel?.();
    this.states.clear();
  }

  update(names: string[], onMs: number, offMs: number): void {
    const above = new Set(names);
    for (const name of new Set([...this.states.keys(), ...above])) {
      let state = this.states.get(name);
      if (!state) {
        state = { above: true, latched: false, since: this.clock.now(), deadline: null, cancel: null };
        this.states.set(name, state);
      }
      const nextAbove = above.has(name);
      if (state.above !== nextAbove) {
        state.cancel?.();
        state.cancel = null;
        state.deadline = null;
        state.above = nextAbove;
        state.since = this.clock.now();
      }
      if (!state.above && !state.latched) {
        state.cancel?.();
        this.states.delete(name);
        continue;
      }
      if (state.above && state.latched) continue;

      const deadline = state.since + (state.above ? onMs : offMs);
      if (state.deadline === deadline) continue;
      state.cancel?.();
      state.deadline = deadline;
      const pending = state;
      state.cancel = this.clock.schedule(Math.max(0, deadline - this.clock.now()), () => {
        // A canceled callback must not survive a media change or timer restart.
        if (this.states.get(name) !== pending || pending.deadline !== deadline) return;
        pending.deadline = null;
        pending.cancel = null;
        if (pending.above) {
          pending.latched = true;
          this.emit(name);
        } else {
          this.states.delete(name);
        }
      });
    }
  }
}

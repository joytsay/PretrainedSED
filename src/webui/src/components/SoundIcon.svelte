<script lang="ts">
  export let kind: string = 'audio';
  export let size = 34;
  export let strokeWidth = 1.8;
</script>

<svg width={size} height={size} viewBox="0 0 48 48" fill="none" stroke="currentColor"
  stroke-width={strokeWidth} stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  {#if kind === 'siren'}
    <path fill="currentColor" stroke="none" d="M15 34V22.5C15 15 18.6 11 24 11s9 4 9 11.5V34H15Z" />
    <path fill="currentColor" stroke="none" d="M12 34h24v4H12zM9.5 38h29a2 2 0 0 1 2 2v2h-33v-2a2 2 0 0 1 2-2Z" />
    <path d="M19 30v-8c0-4 2-6.5 5.5-7.3" stroke="var(--bg, #020b19)" stroke-width="2.6" opacity=".55" />
    <path d="M24 7V2M10.5 12 7 8.5M37.5 12 41 8.5M9 23H3.5M39 23h5.5" stroke-width="3.4" />
  {:else if kind === 'violence'}
    <path fill="currentColor" stroke="none" d="M7 15.5 36 11l6 1.5 1 6-3 4-16 3-1 5.5-4 4-1 10H7l1.5-17-3-2.5Z" />
    <path d="M12 20.5h5M12.5 25h2M21 26c2.5 1.5 5 1 6.5-1.5" stroke="var(--bg, #020b19)" stroke-width="2.5" opacity=".7" />
  {:else if kind === 'vehicle'}
    <path fill="currentColor" stroke="none" d="M4 25h6l6-10h17l7 10h4v13h-5.5a8 8 0 0 0-15.5 0h-7.5a8 8 0 0 0-15.5 0H4V25Z" />
    <circle cx="12" cy="37" r="6" fill="currentColor" stroke="none" />
    <circle cx="35" cy="37" r="6" fill="currentColor" stroke="none" />
    <path fill="var(--bg, #020b19)" stroke="none" d="m18 18-4 7h9v-7h-5ZM26 18v7h10l-5-7h-5Z" />
    <circle cx="12" cy="37" r="2.2" fill="var(--bg, #020b19)" stroke="none" opacity=".45" />
    <circle cx="35" cy="37" r="2.2" fill="var(--bg, #020b19)" stroke="none" opacity=".45" />
  {:else if kind === 'glass'}
    <path d="M5 8h17l-2 13 7 5-7 5-2 10-8-5-7 1 3-9-4-6h8L8 8M28 8h16l-5 9 2 9-9 3-5-3 5-8-4-10Z" stroke-width="2.8" />
    <path d="m10 22 6 3-5 5 9 1M32 18l5 3-5 8M27 26l-6 8 4 7" stroke-width="2.8" />
  {:else if kind === 'impact'}
    <circle cx="9.5" cy="37" r="7" stroke-width="3" />
    <circle cx="38.5" cy="37" r="7" stroke-width="3" />
    <path d="m9.5 37 9-14h9l11 14M17 37h10l-8.5-14M27 23l-5 14M28 18l6-1 3-5M31 12h7" stroke-width="3" />
    <path d="M17 9c5-1 9 1 12 5l4 7M14 7c7-2 13 0 17 5" stroke-width="3" />
    <path d="m15 17 5 3-3 5M5 34l5 3-4 4M34 34l5 3-4 4" stroke-width="2" />
  {:else if kind === 'explosion'}
    <path fill="currentColor" stroke="none" fill-rule="evenodd" d="M24 1 29 18l9-9-2 12 10-4-5 10 7 5-9 3 5 8H31l-1-6-3 1-1-8-3 5-4-5-1 8-4-1-1 6H2l6-8-6-4 9-4-6-8 13 3-1-13 6 12L24 1Zm0 28-3 7-4-2-1 9h16l-1-9-4 2-3-7Z" clip-rule="evenodd" />
  {:else if kind === 'animal'}
    <path fill="currentColor" stroke="none" d="M10 17C13 7 21 2 30 5c7 2 11 9 10 18 5-1 9 1 8 6-1 6-7 10-14 9-3 7-10 11-17 9-6-2-7-7-5-12-5 2-12 0-14-5-2-5 2-11 8-13Z" />
    <path fill="currentColor" stroke="none" d="M2 23c-4 5-1 12 6 13 7 1 14-1 18-6l-3-8-11-4-10 5Z" />
    <ellipse cx="17" cy="17" rx="1.7" ry="2.5" fill="var(--bg, #020b19)" stroke="none" opacity=".65" />
    <ellipse cx="31" cy="16" rx="1.6" ry="2.3" fill="var(--bg, #020b19)" stroke="none" opacity=".65" />
    <path d="M26 25c0 7-5 11-12 10M37 24c0 6-3 10-7 13" stroke="var(--bg, #020b19)" stroke-width="2.4" opacity=".62" />
  {:else}
    <path d="M5 24h5l3-9 5 18 5-25 5 31 5-20 3 10 3-5h4" />
  {/if}
</svg>

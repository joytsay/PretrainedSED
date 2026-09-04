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
    <path fill="currentColor" stroke="none" d="M3 27c0-2 1.5-3.5 4-4l6-1 5-8h12l5 8 6 1c2.5.5 4 2 4 4v8c0 2-1 3-3 3h-1a7 7 0 0 0-10-6 7 7 0 0 0-2 6H19a7 7 0 0 0-14 0H3V27Z" />
    <circle cx="12" cy="37" r="6" fill="currentColor" stroke="none" />
    <circle cx="36" cy="37" r="6" fill="currentColor" stroke="none" />
    <path fill="var(--bg, #020b19)" stroke="none" d="m19 17-4 6h8v-6h-4ZM26 17v6h7l-4-6h-3Z" />
    <path d="M4 29h4M40 29h4" stroke="var(--bg, #020b19)" stroke-width="1.8" opacity=".45" />
    <circle cx="12" cy="37" r="2.2" fill="var(--bg, #020b19)" stroke="none" opacity=".45" />
    <circle cx="36" cy="37" r="2.2" fill="var(--bg, #020b19)" stroke="none" opacity=".45" />
  {:else if kind === 'glass'}
    <path d="M5 8h17l-2 13 7 5-7 5-2 10-8-5-7 1 3-9-4-6h8L8 8M28 8h16l-5 9 2 9-9 3-5-3 5-8-4-10Z" stroke-width="2.8" />
    <path d="m10 22 6 3-5 5 9 1M32 18l5 3-5 8M27 26l-6 8 4 7" stroke-width="2.8" />
  {:else if kind === 'impact'}
    <path fill="currentColor" stroke="none" d="M21.2 3.8a3.2 3.2 0 0 1 5.6 0l20.7 37.4A3.2 3.2 0 0 1 44.7 46H3.3a3.2 3.2 0 0 1-2.8-4.8L21.2 3.8Z" />
    <path fill="var(--bg, #020b19)" stroke="none" d="M24 8 5.8 41h36.4L24 8Z" />
    <path fill="currentColor" stroke="none" d="M24 12.2 9.5 38h29L24 12.2Z" />
    <path fill="var(--bg, #020b19)" stroke="none" d="m25.5 17-7 14h6l-2 10 9-16h-5.8l3-8h-3.2Z" />
  {:else if kind === 'explosion'}
    <path fill="currentColor" stroke="none" fill-rule="evenodd" d="M24 1 29 18l9-9-2 12 10-4-5 10 7 5-9 3 5 8H31l-1-6-3 1-1-8-3 5-4-5-1 8-4-1-1 6H2l6-8-6-4 9-4-6-8 13 3-1-13 6 12L24 1Zm0 28-3 7-4-2-1 9h16l-1-9-4 2-3-7Z" clip-rule="evenodd" />
  {:else if kind === 'animal'}
    <path fill="currentColor" stroke="none" d="M.2 38.8c2.1-5.6 6.5-13.8 6.2-17.1C3.3 17.5 2.1 12 5.4 8.8 9.9 5.1 14 1.3 18.8.2c8.6-1.1 16.9 3.2 18.4 11.1.8 3.6 3.3 4.9 8.2 6.4 2.6.8 3.3 2 2.4 4.9-1.6 5.3-4.8 8.6-9.3 9.9-3.8 1-7.6.3-10.8.4-2.9.1-3.8 2.2-3 5.2 1 3.9 3.8 9.5 3.3 11.8-.4 2-2 1-3.5-.2-4.2-3.6-8.8-7.3-14.5-9.1-4.8-1.5-8.4-.3-13.2.3-2.7.4-4.2.2-3.6-2.1Z" />
    <path d="M21.2 4.1c-2.2 3.1-1.9 8.4-1.8 13.3.2 7.4-.7 10.4-3.8 11.8-4.9 2.2-10.8-.8-12.9-6.2" fill="none" stroke="var(--bg, #020b19)" stroke-width="2.5" />
  {:else}
    <path d="M5 24h5l3-9 5 18 5-25 5 31 5-20 3 10 3-5h4" />
  {/if}
</svg>

# Development website review package

Target: https://behindtheveil369.work. Canonical source revision inspected: dadc445e4f6dc1efeae58f943134ca48d54adb80. Existing Sites development ledger project preserved. No source push or deployment occurred.

At discovery, there was no dedicated ANNA page. The patch adds /anna, /anna-neural, local evidence assets and links from the existing home project list. The ANNA page distinguishes orchestration from the trained neural project. All listed model metrics come from the completed local evidence, not placeholders. GitHub link remains clearly pending account selection/publication approval.

The review patch is demonstrations/website-neural-forge.patch. It includes the new pages, responsive CSS, measured SVG plots, benchmark CSV/JSON, markdown documentation, four route tests and a targeted TypeScript QA configuration. No existing AETERNA, Concordia or Samara implementation was changed. Existing public routes, game builds and tests pass. Broad root TypeScript retains exactly 184 pre-existing diagnostics from the original revision. See docs/VERIFICATION.md.

Only after review, obtain the current canonical Site source and check whether its base still matches. If it changed, reconcile the patch against the new source rather than overwriting it. Verify with:

```bash
git apply --check /absolute/path/website-neural-forge.patch
git apply /absolute/path/website-neural-forge.patch
npm test
node --test tests/*.test.mjs
node node_modules/typescript/bin/tsc -p tsconfig.neural-qa.json --noEmit
```

Real desktop/mobile browser checks and accessibility testing remain open. Supported managed-browser capability was unavailable during this task. CSS includes a 600px breakpoint, bounded responsive width, overflow containment for tables, visible keyboard focus, semantic headings and image descriptions; static checks do not substitute for device QA.

The website build uses vinext/Cloudflare Workers with an existing Concordia Vite sub-build. Preserve its D1 binding and hosting identity. Use the Sites workflow for any later approved source synchronization/version/deployment. User authorization is required before deployment; this review package grants none. No repository URL or paid-service wiring is invented.

Source-only authoring is complete. Public release, GitHub link, hardware/device validation and future ANNA integration remain separate decisions. The standalone engineering presentation and package are private review deliverables, not automatically served by this patch.

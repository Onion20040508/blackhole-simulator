# Building a Black Hole Simulator with a Coding Agent
### A case study for "Agents and How to Use Them"

This document reconstructs the complete development of a real-time GPU black-hole
ray tracer (Kerr + Schwarzschild) built entirely through conversation with a
coding agent (Claude Code). It is written to be mined for presentation slides:
each section has the *what happened* and the *agent lesson* underneath.

---

## 0. One-paragraph summary

Starting from a single sentence — *"I want to simulate and visualize static and
rotating black holes on my 7800X3D + 4080 Super"* — a physics researcher and a
coding agent built, over one long session, a real-time general-relativistic ray
tracer with: gravitational lensing, volumetric accretion disks made of up to
millions of geodesic particles, relativistic jets, tidal disruption events,
self-gravitating stars, reflective spheres, a free-fly camera that crosses the
event horizon, HDR/bloom post-processing, a packaged Windows `.exe`, and a
one-command macOS build. The human never wrote a line of code or ran a command
themselves — they steered with screenshots and physics intuition; the agent
wrote the code, ran it, looked at its own output, and iterated.

**Final artifact:** ~1800 lines of Python/Taichi, two companion analysis
scripts, a packaged distributable, and a Mac port — none of it hand-typed by the
human.

---

## 1. The development timeline (what was built, in order)

| Phase | Human asked for | Agent delivered |
|---|---|---|
| **Kickoff** | "Simulate static + rotating black holes" | Asked 2 scoping questions, then set up a Python 3.11 venv, installed Taichi/CUDA stack, wrote the first ray tracer + a separate f64 analysis script |
| **First light** | (implicit: make it look right) | Rendered a test frame, *looked at it*, saw it was too dark → too white, tuned exposure |
| **Art pass** | "Make UI bigger, better lighting, game-like ray tracing" | HDR + bloom pipeline, volumetric disk, Doppler beaming toggle, real Milky Way sky texture, turbulence |
| **Pre-render** | "Can I pre-render ultra quality then explore?" | Orbit pre-render mode + a separate interactive viewer; explained why free-flight pre-render is physically impossible (5D light field) |
| **Physical realism** | "Jet and disk look bad, I want real physics + WASD camera" | Novikov–Thorne relativistic disk (Page–Thorne flux), M87-style parabolic jet, free-fly camera |
| **Particles** | "Make disk and jet out of particles, adjustable count" | Particle dynamics → density grid → volumetric render bridge |
| **Dynamics** | "I want to watch it evolve; let me place stars" | True geodesic particles with adiabatic inspiral; circularization dissipation |
| **TDE / mergers / inside horizon** | "Can we do mergers, star-eating? What happens inside?" | Built tidal-disruption stars; gave honest feasibility analysis on mergers (no) |
| **Launching** | "Let me *shoot* stars, with a prediction line" | Right-click aim with a live geodesic prediction trajectory, color-coded by fate |
| **Big refactor** | "Higher limits, delete escaped particles, 3D grid background, rigid body, packaging?" | Empty-space skipping, particle GC, mirror ball, occupancy grid |
| **Perf crisis** | "It's way slower than before" | Diagnosed template-recompile thrash; dual-kernel design (lean vs full) |
| **Polish** | "UI is ugly, buttons invisible, remove clutter" | Iterated UI 4+ times (tkinter → ttk → in-window imgui), fixed contrast |
| **Physics features** | "Self-gravity? Temperature → color? Mass?" | Plummer self-gravity, friction-heating temperature model, mass-via-radius |
| **Bug hunts** | "Why white stars? Why jets from the center? Why staircase aliasing?" | Each traced to a *specific* root cause (tonemap clip / f32 mass-shell drift / voxel resolution) |
| **Ship it** | "Package it. Also Mac." | PyInstaller Windows build + zip; a `build_mac.sh` one-command Mac port |
| **The crash** | "It won't open anymore" | Multi-hour root-cause hunt → **corrupted CUDA driver state**, not the code; added a self-healing Vulkan fallback |

---

## 2. The core interaction patterns (the part for your slides)

### Pattern A — Scope with questions, then act autonomously
The very first move was **not** to write code. The agent asked two structured
questions (real-time vs offline? which physics?) and then proceeded for dozens of
steps without further hand-holding: created the venv, picked Python 3.11 (because
the system Python 3.14 had no Taichi wheels — a decision the agent made and
explained), installed dependencies in the background, wrote ~600 lines, and
test-rendered.

> **Lesson:** Good agents front-load the *decisions that change the outcome*, then
> stop asking permission for the obvious mechanical steps. The human answered two
> dropdowns and got a working renderer.

### Pattern B — The visual feedback loop (this is the headline)
The agent didn't just write rendering code and hope. It **rendered a frame, read
the PNG back, and looked at it** — repeatedly, throughout the whole project. When
the first frame came out too dark, then blown-out white, the agent saw that in
the image and tuned exposure. When the accretion disk had "staircase" aliasing,
the agent rendered it, saw the blocky voxels, and traced it to grid resolution.

> **Lesson:** An agent that can *observe its own output* (screenshots, logs,
> test results) closes the loop without the human in it. The human became a
> second pair of eyes for *taste* ("this looks too white"), not a debugger.

### Pattern C — Long-running work goes to the background
Dependency installs, 8K renders (23040×12960, ~300M rays), and PyInstaller builds
were launched as background tasks so the conversation kept moving. The agent got
a completion notification and picked the result back up.

> **Lesson:** Agents should not block on slow work. Fire-and-notify keeps the
> human's turn-around fast.

### Pattern D — Steering by artifact, not by instruction
The human's most common input was **a screenshot with a circle drawn on it** and
a one-line complaint: "why is the grid line fuzzy?", "the stars are too rough",
"buttons are invisible". They almost never specified *how* to fix anything. The
agent supplied the diagnosis and the fix.

> **Lesson:** The human operates at the level of *intent and judgment*; the agent
> owns *mechanism*. This is the natural division of labor and it scales.

### Pattern E — Honest "no" and honest tradeoffs
Multiple times the agent declined or qualified instead of blindly building:
- **Blender/UE5 for better lighting?** → "No, game engines can't bend light;
  don't go there," with the physics reason.
- **Pre-render a free-flight experience?** → "That's a 5-D light field, storage is
  astronomical; here's the achievable compromise (orbit pre-render)."
- **Let the star evolve into a disk by self-gravity alone?** → "Collisionless
  particles never circularize on their own; real TDEs need stream collisions —
  so I added a dissipation term."
- **Use this to test your decoherence research?** → "Quantitatively no (this is
  classical geometric optics; your effect lives in the soft-photon regime it
  can't see); here's what it *can* honestly help with."

> **Lesson:** The most valuable agent behavior in a research setting is calibrated
> honesty. A yes-machine would have wasted days in Blender and produced a
> physically wrong "decoherence test."

### Pattern F — Root-cause discipline over pattern-matching
Three "bugs" looked like rendering glitches but had distinct real causes, and the
agent found each one rather than slapping on a visual band-aid:
- **White stars** → HDR values clipped past the tonemapper's white point →
  color-preserving luminance cap.
- **Jets shooting from the center** → *not* a feature; f32 integration error in
  the strong-field zone broke the mass-shell condition, ejecting particles
  superluminally → strong-field micro-stepping + mass-shell reprojection + a
  superluminal-garbage delete.
- **Staircase aliasing** → fixed 320³ grid over a huge box meant 0.35 M voxels →
  adaptive grid that auto-fits the particle bounding box (≈3× finer).

> **Lesson:** "Looks like X" is a hypothesis, not a diagnosis. The agent
> instrumented, bisected, and confirmed before fixing.

---

## 3. Case study: the closing crash (a whole slide on its own)

**Symptom:** After a working session, the app stopped opening. Exit code
`0xC0000005` (native access violation), **no Python traceback**.

**The trap:** It pattern-matched to "my code broke." The crash point even seemed
to move depending on resolution and parameters — tempting to "fix" the last thing
edited.

**The discipline:** The agent bisected systematically:
1. Confirmed it crashed in *kernel warm-up*, not the main loop.
2. Bisected the giant render kernel block-by-block (replaced compile-time flags,
   stubbed sections) — every variant still crashed.
3. Tried CPU backend → worked. Disabled the disk cache → still crashed on CUDA.
4. Wrote a **15-line trivial CUDA kernel** → it *also* crashed, 3 times out of 3.
5. Same trivial kernel on the **Vulkan backend** → worked perfectly.

**Root cause:** Not the code at all. Days of testing had hard-crashed dozens of
CUDA processes, progressively **corrupting the driver's JIT state** until even a
trivial kernel couldn't compile. The "moving crash point" was a red herring —
it was just whether a given kernel was already in the disk compile-cache.

**The fix that mattered:** The agent reverted all the speculative "fixes" it had
made under the wrong assumption (a big-stack thread hack, `real_func`
refactor, disabling the optimizer), then added a **sacrificial-subprocess CUDA
self-probe**: at startup the app compiles a tiny CUDA kernel in a throwaway
process; if that crashes, the main app automatically falls back to Vulkan and
keeps running. The human's actual remedy: **reboot** (restores driver state).

> **Lesson for the talk:** This is the single best agent-behavior story in the
> project. Before "fixing" something, verify the evidence supports *that specific
> cause*. A signal that pattern-matches to a known failure can have a completely
> different origin. The agent also left the system more robust than before
> (auto-fallback), not just patched.

---

## 4. What the agent handled that a human typically context-switches for

- **Environment archaeology:** detected Python 3.14 had no Taichi wheels, found
  3.11, built an isolated venv, kept the venv working after moving the project
  folder.
- **Numerical methods:** chose Kerr–Schild coordinates (horizon-penetrating),
  derived analytic metric gradients, picked RK4 then later RK2 with an
  adaptive bending-angle-capped step, built a Novikov–Thorne flux lookup table by
  numerically integrating Page–Thorne.
- **GPU performance engineering:** empty-space skipping via an occupancy mip,
  template specialization (and then *un*-specialization when it caused recompile
  thrash), adaptive voxel grids, dual lean/full kernels.
- **Graphics pipeline:** HDR buffer → bloom → starburst → ACES tonemap → gamma →
  upscale, plus temporal accumulation for "converge-when-still" quality.
- **Packaging & ports:** PyInstaller bundle (including the non-obvious trick of
  re-bundling the source for Taichi's JIT), a cross-platform Mac build script with
  backend auto-fallback and platform-tuned defaults.
- **Honest physics consulting:** redshift factors, ISCO/ergosphere/photon-orbit
  formulas, tidal radius, why mergers need numerical relativity, why the tool
  can't quantify horizon decoherence.

---

## 5. The division of labor (a clean two-column slide)

| The human (researcher) provided | The agent provided |
|---|---|
| The goal and the hardware | All code, all commands, all builds |
| Physics intuition & taste | Numerical methods & GR formulas |
| Screenshots + one-line critiques | Diagnosis + implementation |
| Priorities ("do this, not that") | Tradeoff analysis & honest "no"s |
| The reboot at the very end | Everything that led to discovering it needed one |

The human wrote **zero lines of code** and ran **zero commands**.

---

## 6. Numbers for a stats slide

- **~1,800 lines** of Python/Taichi in the main file, plus 2 companion scripts.
- **~30 distinct feature/fix requests**, each a short message — most under a
  sentence, several just an annotated screenshot.
- **Particles:** scaled from 0 → adjustable up to **millions** of geodesic
  particles, each integrated on the GPU every frame.
- **Render scale:** from a 640×360 test frame to **8K (7680×4320) with 3×3
  supersampling** (~300 million primary rays) for wallpapers.
- **Backends supported by the end:** CUDA, Vulkan (auto-fallback), CPU (debug),
  Metal (Mac).
- **Platforms shipped:** Windows `.exe` + zip, macOS one-command build.

---

## 7. Takeaways to end the presentation on

1. **Scope by question, execute by default.** Ask the few things that change the
   outcome; then act without narrating every step.
2. **Give the agent eyes.** The screenshot-read-back loop is what made
   visual/physical iteration possible without the human debugging.
3. **Steer by artifact and intent.** "This looks wrong" + a screenshot is a
   complete, efficient instruction to a capable agent.
4. **Value the honest "no."** Refusing Blender, refusing a fake decoherence test,
   and qualifying the self-gravity model saved more time than any feature.
5. **Diagnose, don't pattern-match.** The closing crash was a driver problem
   masquerading as a code bug; discipline beat the obvious-but-wrong fix.
6. **Leave it more robust than you found it.** The fix wasn't "make it work once"
   — it was an auto-detecting fallback so it never hard-fails that way again.

---

*Project: `Physics/blackhole-sim/` — a Taichi/CUDA Kerr–Schwarzschild ray tracer.
Built conversationally with Claude Code. This document was generated by the agent
on request, recalling the full session for presentation use.*

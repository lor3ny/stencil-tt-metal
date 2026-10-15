# jacobi_blackhole

5-point Jacobi solver for the 2-D Laplace equation in bf16, ported from `../jacobi_optimised`
(Grayskull e150, tt-metal of August 2024) to Blackhole with tt-metal `1f2ae9b5` (5 September 2026).

| File | Role |
|---|---|
| `jacobi.cpp` | Host: Mesh API setup, work split, CPU reference and comparison, timing |
| `kernels/dataflow/stream_in.cpp` | Reader on RISCV_1 / NoC 1: sweep barrier, double-buffered 34-row windows, neighbour pages |
| `kernels/compute/laplace.cpp` | Compute: `E + W + S + N` in the DST register, `x 0.25` on the SFPU |
| `kernels/dataflow/stream_out.cpp` | Writer on RISCV_0 / NoC 0: result rows to DRAM, end-of-sweep signals to neighbours |

## Build

Needs tt-metal `1f2ae9b5`, built and installed with its `metalium-dev`, `metalium-runtime` and
`jit-build` components and the dependencies they export (fmt, spdlog, nlohmann_json, umd, tt-logger).

```sh
cmake -B build -G Ninja -DCMAKE_PREFIX_PATH=<tt-metal install prefix>
cmake --build build
```

It also builds inside a tt-metal tree through `add_subdirectory`, like the programming examples.

## Run

```sh
export TT_METAL_RUNTIME_ROOT=<tt-metal install prefix>/libexec/tt-metalium   # or the tt-metal source tree
./build/jacobi X Y ITS CX CY [--no-verify]
```

- `X`, `Y`: interior grid size, multiples of 32. `ITS`: number of sweeps.
- `CX`, `CY`: cores along the rows and the columns. Every core needs at least one 32 x 32 batch
  (`CX <= X/32`, `CY <= Y/32`), and `CX x CY` must fit the compute grid the program prints.
- The CPU reference runs by default; the program exits with status 1 if any grid point differs by more
  than 1e-3 + 2^-6 x |reference|. Use `--no-verify` to skip it for large benchmark runs.
- Kernels are compiled at the first launch, from the `kernels/` directory recorded at build time.
  `JACOBI_KERNEL_DIR` overrides that location.
- On a new board, run once with `TT_METAL_WATCHER=1` to have tt-metal check NoC addresses and alignment.

The timings exclude kernel compilation (a one-sweep warm-up launch runs first), and the upload time
now waits until the data is in DRAM.

## Changes from jacobi_optimised

- The prefetch now reads batch k + 1 into the free L1 half while batch k is copied (it used to copy
  the wrong half).
- Cores wait for their four neighbours between sweeps. Each core keeps five semaphores, counting
  finished sweeps of itself and of its north, south, west and east neighbours.
- Ranks and batch ranges come from the host as runtime arguments, not from NoC coordinates, and one
  program covers the whole grid.
- DRAM reads follow Blackhole's 64-byte alignment. Rows are padded with 32 columns per side, and the
  alignment comes from the HAL, so the code also runs on Wormhole (32 bytes).
- The grids use one DRAM page per padded row, so they are interleaved over all eight banks.
- The three additions stay in the DST register (fp32) and the scale by 0.25 runs on the SFPU, so each
  batch is unpacked and packed once instead of four times.
- The comparison against the CPU reference is back on, and the point count uses 64-bit arithmetic.

Not yet run on Blackhole hardware. The code compiles against tt-metal `1f2ae9b5`, the kernels build for
Blackhole on a mock device, and the program matches the CPU reference bit for bit under tt-emule.

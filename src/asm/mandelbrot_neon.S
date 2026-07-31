/* ==========================================================================
 *  mandelbrot_neon.S — Kernel C': hand-rolled AArch64 NEON Mandelbrot kernel
 * ==========================================================================
 *
 *  ABI: AAPCS64.
 *
 *      void mandelbrot_neon_asm(uint16_t*            out,        // x0
 *                               const KernelParams*  p,          // x1
 *                               int32_t              row_begin,  // w2
 *                               int32_t              row_end);   // w3
 *
 *  The arm64 counterpart to mandelbrot_avx2.asm, and the reason this suite
 *  runs natively on Apple Silicon. Same structure, same algebra, 4 lanes
 *  instead of 8.
 *
 *  Deliberate properties:
 *
 *    * True leaf function: no stack frame, no frame pointer, no callee-saved
 *      register touched. Only x0-x10 and v0-v7/v16-v23 are used, all of which
 *      are caller-saved under AAPCS64. Note v8-v15 are avoided entirely
 *      because their low 64 bits *are* callee-saved.
 *    * No FMLA, for the same reason the AVX2 kernel avoids FMA: zx^2 and zy^2
 *      are both needed separately, and a fused form would break bit-exactness
 *      with the scalar C++ reference.
 *    * Written as .S (not .s) so the C preprocessor handles the Mach-O vs ELF
 *      differences below.
 *
 *  Assembled by clang's integrated assembler; NASM is x86-only.
 * ========================================================================== */

/* Mach-O prefixes C symbols with an underscore; ELF does not. */
#if defined(__APPLE__)
#  define CSYM(name) _##name
#else
#  define CSYM(name) name
#endif

/* Operand spellings for a PC-relative page + page-offset symbol reference.
 *
 * These are deliberately two separate operand macros rather than one macro
 * emitting both instructions: a multi-instruction macro would need a statement
 * separator, and `;` is treated as a comment on this target, which silently
 * drops everything after it. That failure mode is invisible (it still
 * assembles) and produces a garbage constant, so it is worth avoiding by
 * construction. */
#if defined(__APPLE__)
#  define PAGE_HI(sym) sym@PAGE
#  define PAGE_LO(sym) sym@PAGEOFF
#else
#  define PAGE_HI(sym) sym
#  define PAGE_LO(sym) :lo12:sym
#endif

/* --- struct KernelParams byte offsets (mirrored by static_assert in kernel_api.hpp) */
#define P_CX0         0
#define P_CY0         4
#define P_DX          8
#define P_DY         12
#define P_ESCAPE_R2  16
#define P_VEC_WIDTH  20
#define P_HEIGHT     24
#define P_MAX_ITER   28
#define P_STRIDE     32
#define P_WIDTH      36

/* ---------------------------------------------------------------------------
 *  Read-only constant
 * ------------------------------------------------------------------------- */
#if defined(__APPLE__)
    .section __TEXT,__const
#else
    .section .rodata
#endif
    .p2align 4
Llane_offsets:
    .float 0.0, 1.0, 2.0, 3.0

/* ---------------------------------------------------------------------------
 *  Register map
 * ---------------------------------------------------------------------------
 *   x0   running row pointer (advances by row_bytes each scanline)
 *   x1   const KernelParams*
 *   x2   y  (current scanline)
 *   x3   row_end
 *   x4   vec_width
 *   x5   row_bytes  (stride * sizeof(uint16_t))
 *   w6   max_iter
 *   w7   inner iteration countdown
 *   x9   scratch (constant addressing, mask reduction, store address)
 *   x10  x  (pixel column, always a multiple of 4)
 *
 *   v0  zx        v16  escape_r2      v20  cx0
 *   v1  zy        v17  lane offsets   v21  cy0
 *   v2  cx        v18  iter (int32x4) v22  dy
 *   v3  cy        v19  dx             v23  scratch
 *   v4  zx^2   v5  zy^2   v6  magnitude/scratch   v7  escape mask
 * ------------------------------------------------------------------------- */
    .text
    .p2align 4
    .globl CSYM(mandelbrot_neon_asm)
#if !defined(__APPLE__)
    .type CSYM(mandelbrot_neon_asm), %function
#endif

CSYM(mandelbrot_neon_asm):
    cmp     w2, w3                      // row_begin >= row_end -> nothing to do
    b.ge    Lepilogue

    ldr     w4, [x1, #P_VEC_WIDTH]
    cmp     w4, #0                      // zero-width image -> nothing to do
    b.le    Lepilogue

    sxtw    x2, w2                      // y       = row_begin
    sxtw    x3, w3                      // row_end
    sxtw    x4, w4                      // vec_width
    ldr     w6, [x1, #P_MAX_ITER]
    ldr     w5, [x1, #P_STRIDE]         // 32-bit load zero-extends
    lsl     x5, x5, #1                  // row_bytes = stride * 2

    // Bias the output pointer to the first requested scanline.
    madd    x0, x2, x5, x0              // x0 = out + y * row_bytes

    // Hoist loop-invariant broadcasts. ld1r has no immediate-offset form, so
    // each address is materialised into x9 first.
    ld1r    { v20.4s }, [x1]            // cx0        (offset 0)
    add     x9, x1, #P_CY0
    ld1r    { v21.4s }, [x9]            // cy0
    add     x9, x1, #P_DX
    ld1r    { v19.4s }, [x9]            // dx
    add     x9, x1, #P_DY
    ld1r    { v22.4s }, [x9]            // dy
    add     x9, x1, #P_ESCAPE_R2
    ld1r    { v16.4s }, [x9]            // escape_r2
    adrp    x9, PAGE_HI(Llane_offsets)
    add     x9, x9, PAGE_LO(Llane_offsets)
    ldr     q17, [x9]                   // [0.0, 1.0, 2.0, 3.0]

/* ------------------------------------------------------------------------- */
Lrow_loop:
    // cy = cy0 + (float)y * dy  — one multiply, one add, matching scalar C++.
    scvtf   s3, w2
    dup     v3.4s, v3.s[0]
    fmul    v3.4s, v3.4s, v22.4s
    fadd    v3.4s, v3.4s, v21.4s

    mov     x10, #0                     // x = 0

/* ------------------------------------------------------------------------- */
Lblock_loop:
    // cx = cx0 + ((float)x + lane) * dx.
    // (x + lane) is formed before the multiply to preserve bit-exactness with
    // the scalar kernel; hoisting cx0 + x*dx would round differently.
    scvtf   s23, w10
    dup     v23.4s, v23.s[0]
    fadd    v23.4s, v23.4s, v17.4s
    fmul    v23.4s, v23.4s, v19.4s
    fadd    v2.4s,  v23.4s, v20.4s

    movi    v0.4s, #0                   // zx   = 0
    movi    v1.4s, #0                   // zy   = 0
    movi    v18.4s, #0                  // iter = 0

    mov     w7, w6                      // n = max_iter
    cbz     w7, Lstore                 // max_iter == 0 -> store zeros

/* -------------------------------------------------------------------------
 *  Inner recurrence: z <- z^2 + c, counting steps whose entry magnitude is
 *  <= escape_r2.
 */
Liter_loop:
    fmul    v4.4s, v0.4s, v0.4s         // zx^2
    fmul    v5.4s, v1.4s, v1.4s         // zy^2
    fadd    v6.4s, v4.4s, v5.4s         // |z|^2

    // fcmge is a quiet compare: (escape_r2 >= |z|^2) is exactly |z|^2 <= r2,
    // and escaped lanes holding inf/NaN yield false and latch, matching
    // _CMP_LE_OQ on x86.
    fcmge   v7.4s, v16.4s, v6.4s        // mask = |z|^2 <= escape_r2

    umaxv   s23, v7.4s                  // horizontal OR-equivalent for a
    fmov    w9, s23                     //   0 / 0xFFFFFFFF lane mask
    cbz     w9, Lstore                 // whole batch has escaped

    sub     v18.4s, v18.4s, v7.4s       // mask lane == -1 -> iter += 1

    fmul    v23.4s, v0.4s, v1.4s        // t  = zx*zy
    fadd    v23.4s, v23.4s, v23.4s      // t  = 2*zx*zy
    fadd    v1.4s,  v23.4s, v3.4s       // zy = 2*zx*zy + cy
    fsub    v6.4s,  v4.4s, v5.4s        // zx^2 - zy^2
    fadd    v0.4s,  v6.4s, v2.4s        // zx = zx^2 - zy^2 + cx

    subs    w7, w7, #1
    b.ne    Liter_loop

/* -------------------------------------------------------------------------
 *  Narrow 4 x int32 -> 4 x uint16 and store 8 bytes. Truncation is lossless
 *  because max_iter <= 65535 is enforced by the caller.
 *
 *  The store address is computed into x9 rather than using [x0, x10, lsl #1]:
 *  a 64-bit register-offset store only permits LSL #0 or #3, never #1.
 */
Lstore:
    xtn     v23.4h, v18.4s
    add     x9, x0, x10, lsl #1
    str     d23, [x9]

    add     x10, x10, #4
    cmp     x10, x4
    b.lt    Lblock_loop

    add     x0, x0, x5                  // advance one scanline
    add     x2, x2, #1
    cmp     x2, x3
    b.lt    Lrow_loop

Lepilogue:
    ret

#if defined(__APPLE__)
    .subsections_via_symbols
#endif

; ===========================================================================
;  mandelbrot_avx2.asm — Kernel C: hand-rolled x86-64 AVX2 Mandelbrot kernel
; ===========================================================================
;
;  ABI: System V AMD64.
;
;      void mandelbrot_avx2_nasm(uint16_t*            out,        ; rdi
;                                const KernelParams*  p,          ; rsi
;                                int32_t              row_begin,  ; edx
;                                int32_t              row_end);   ; ecx
;
;  Deliberate properties:
;
;    * Zero stack frame. Every GPR used (rax, rcx, rdx, rsi, rdi, r8-r11) is
;      caller-saved under System V, so there is no push/pop, no rbp chain and
;      no red-zone traffic. The function is a true leaf.
;    * Pure AVX2 — no FMA. The recurrence needs zx^2 and zy^2 as distinct
;      values for both the escape test and the real update, leaving no
;      multiply-add pair to contract. Staying FMA-free keeps CPUID
;      requirements to AVX2 alone and keeps results bit-identical to the
;      scalar C++ reference.
;    * The escape mask is tested with VPTEST rather than VMOVMSKPS+TEST,
;      which keeps the early-out off the general-purpose register file
;      entirely (all nine caller-saved GPRs are already committed).
;
;  Requires: AVX2. The caller gates entry on CPUID leaf 7 EBX[5] plus
;  OSXSAVE/XGETBV YMM-state checks (see cpu_features.cpp).
; ===========================================================================

bits 64
default rel

%ifidn __?OUTPUT_FORMAT?__, win64
    %error "mandelbrot_avx2.asm targets the System V AMD64 ABI. Win64 passes args in rcx/rdx/r8/r9 and requires XMM6-15 preservation; build this file for elf64 or macho64."
%endif

; Mach-O prefixes C symbols with an underscore; ELF does not.
%ifidn __?OUTPUT_FORMAT?__, macho64
    %define CSYM(name) _ %+ name
%else
    %define CSYM(name) name
%endif

; --- struct KernelParams byte offsets (mirrored by static_assert in kernel_api.hpp)
%define P_CX0        0
%define P_CY0        4
%define P_DX         8
%define P_DY        12
%define P_ESCAPE_R2 16
%define P_VEC_WIDTH 20
%define P_HEIGHT    24
%define P_MAX_ITER  28
%define P_STRIDE    32
%define P_WIDTH     36

; VCMPPS predicate: LE_OQ (less-or-equal, ordered, non-signalling).
; Matches _CMP_LE_OQ used by the intrinsics kernel. Escaped lanes carrying
; +inf or NaN compare false and stay false, so the mask latches for free.
%define CMP_LE_OQ 0x12

; ---------------------------------------------------------------------------
;  Register map
; ---------------------------------------------------------------------------
;   rdi  running row pointer (advances by row_bytes each scanline)
;   rsi  const KernelParams*
;   rax  max_iter
;   rcx  inner iteration countdown
;   rdx  x  (pixel column, always a multiple of 8)
;   r8   y  (current scanline)
;   r9   row_end
;   r10d vec_width
;   r11  row_bytes  (stride * sizeof(uint16_t))
;
;   ymm0  zx          ymm8   iter accumulator (int32 x8)
;   ymm1  zy          ymm9   escape mask
;   ymm2  cx          ymm10  lane offsets [0..7] (float)
;   ymm3  cy          ymm11  dx broadcast
;   ymm4  zx^2        ymm12  scratch
;   ymm5  zy^2        ymm13  dy broadcast
;   ymm6  magnitude / scratch
;   ymm7  escape_r2   ymm14  cx0 broadcast
;                     ymm15  cy0 broadcast
; ---------------------------------------------------------------------------

section .text

global CSYM(mandelbrot_avx2_nasm)

CSYM(mandelbrot_avx2_nasm):
    cmp     edx, ecx                        ; row_begin >= row_end -> nothing to do
    jge     .epilogue

    mov     r10d, [rsi + P_VEC_WIDTH]
    test    r10d, r10d                      ; zero-width image -> nothing to do
    jle     .epilogue

    movsxd  r8, edx                         ; y        = row_begin
    movsxd  r9, ecx                         ; row_end
    mov     eax, [rsi + P_MAX_ITER]
    mov     r11d, [rsi + P_STRIDE]          ; 32-bit load zero-extends
    shl     r11, 1                          ; row_bytes = stride * 2

    ; Bias the output pointer to the first requested scanline.
    mov     rcx, r8
    imul    rcx, r11
    add     rdi, rcx                        ; rdi = &out[row_begin * stride]

    ; Hoist loop-invariant broadcasts.
    vbroadcastss ymm7,  [rsi + P_ESCAPE_R2]
    vbroadcastss ymm11, [rsi + P_DX]
    vbroadcastss ymm13, [rsi + P_DY]
    vbroadcastss ymm14, [rsi + P_CX0]
    vbroadcastss ymm15, [rsi + P_CY0]
    vmovups      ymm10, [lane_offsets]

; ---------------------------------------------------------------------------
.row_loop:
    ; cy = cy0 + (float)y * dy   — one multiply, one add, matching scalar C++.
    vxorps       xmm3, xmm3, xmm3
    vcvtsi2ss    xmm3, xmm3, r8d
    vbroadcastss ymm3, xmm3
    vmulps       ymm3, ymm3, ymm13
    vaddps       ymm3, ymm3, ymm15

    xor     edx, edx                        ; x = 0 (also clears rdx for indexing)

; ---------------------------------------------------------------------------
.block_loop:
    ; cx = cx0 + ((float)x + lane) * dx.
    ; Forming (x + lane) before the multiply is what preserves bit-exactness
    ; with the scalar kernel; hoisting cx0 + x*dx and adding lane*dx after
    ; would round differently.
    vxorps       xmm12, xmm12, xmm12
    vcvtsi2ss    xmm12, xmm12, edx
    vbroadcastss ymm12, xmm12
    vaddps       ymm12, ymm12, ymm10
    vmulps       ymm12, ymm12, ymm11
    vaddps       ymm2,  ymm12, ymm14

    vxorps  ymm0, ymm0, ymm0                ; zx   = 0
    vxorps  ymm1, ymm1, ymm1                ; zy   = 0
    vpxor   ymm8, ymm8, ymm8                ; iter = 0

    mov     ecx, eax                        ; n = max_iter
    test    ecx, ecx
    jz      .store                          ; max_iter == 0 -> store zeros

; ---------------------------------------------------------------------------
;  Inner recurrence: 9 instructions, one loop-carried dependency chain.
;    z <- z^2 + c,  counting steps whose entry magnitude is <= escape_r2.
.iter_loop:
    vmulps  ymm4, ymm0, ymm0                ; zx^2
    vmulps  ymm5, ymm1, ymm1                ; zy^2
    vaddps  ymm6, ymm4, ymm5                ; |z|^2
    vcmpps  ymm9, ymm6, ymm7, CMP_LE_OQ     ; mask = |z|^2 <= escape_r2

    vptest  ymm9, ymm9                      ; ZF=1 <=> mask is all zero
    jz      .store                          ; whole batch has escaped

    vpsubd  ymm8, ymm8, ymm9                ; mask lane == -1 -> iter += 1

    vmulps  ymm12, ymm0, ymm1               ; t  = zx*zy
    vaddps  ymm12, ymm12, ymm12             ; t  = 2*zx*zy
    vaddps  ymm1,  ymm12, ymm3              ; zy = 2*zx*zy + cy
    vsubps  ymm6,  ymm4, ymm5               ; zx^2 - zy^2
    vaddps  ymm0,  ymm6, ymm2               ; zx = zx^2 - zy^2 + cx

    dec     ecx
    jnz     .iter_loop

; ---------------------------------------------------------------------------
;  Pack 8 x int32 -> 8 x uint16 in lane order and store 16 bytes.
;
;    vpackusdw (src1 == src2) gives, as qwords:
;        q0 = [i0 i1 i2 i3]   q1 = [i0 i1 i2 i3]
;        q2 = [i4 i5 i6 i7]   q3 = [i4 i5 i6 i7]
;    so vpermq with imm 0x08 selects (q0, q2) into the low 128 bits, i.e.
;    i0..i7 contiguous and in order. Saturation is a no-op because
;    max_iter <= 65535 is enforced by the caller.
.store:
    vpackusdw ymm12, ymm8, ymm8
    vpermq    ymm12, ymm12, 0x08
    vmovdqu   [rdi + rdx*2], xmm12

    add     edx, 8
    cmp     edx, r10d
    jl      .block_loop

    add     rdi, r11                        ; advance one scanline
    inc     r8
    cmp     r8, r9
    jl      .row_loop

.epilogue:
    vzeroupper                              ; avoid AVX<->SSE transition stalls
    ret

; ---------------------------------------------------------------------------
;  Read-only constant. Kept in .text (not .rodata) because NASM's macho64
;  backend does not accept .rodata as a section alias; this way one source
;  file assembles unchanged for both elf64 and macho64. Placed after `ret`
;  so it is never reached as code.
    align 32
lane_offsets:
    dd 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0

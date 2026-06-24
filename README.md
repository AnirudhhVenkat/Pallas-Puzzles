# Pallas Puzzles

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rdyro/Pallas-Puzzles/blob/main/Pallas-Puzzles.ipynb)

*Fork of [Triton Puzzles](https://github.com/srush/Triton-Puzzles), adapted for [Pallas](https://jax.readthedocs.io/en/latest/pallas/index.html) in JAX.*

<p align="center">
<img style="max-width:200px;width:100%;max-height:200px" src="pallas_puzzles_logo.svg">
</p>

By Robert Dyro, Ed Schmerling

Interactive puzzles for learning Pallas from first principles. Work through [Pallas-Puzzles.ipynb](Pallas-Puzzles.ipynb); solutions are in [Pallas-Puzzles-Solutions.ipynb](Pallas-Puzzles-Solutions.ipynb).

**Tip:** On Mac, use `interpret=True` in `test(...)` or set `JAX_PLATFORMS=cpu` — Pallas does not support the MPS backend.

## Progress

- [x] Puzzle 1: Constant Add
- [x] Puzzle 2: Constant Add Block
- [ ] Puzzle 3: Outer Vector Add
- [ ] Puzzle 4: Outer Vector Add Block
- [ ] Puzzle 5: Fused Outer Multiplication
- [ ] Puzzle 6: Fused Outer Multiplication - Backwards
- [ ] Puzzle 7: Long Sum
- [ ] Puzzle 8: Long Softmax
- [ ] Puzzle 9: Simple FlashAttention
- [ ] Puzzle 10: Two Dimensional Convolution
- [ ] Puzzle 11: Matrix Multiplication
- [ ] Puzzle 12: Quantized Matrix Mult

## Resources

- [Pallas for Beginners](https://huggingface.co/blog/ariG23498/pallas-for-beginners)
- [Pallas kernel performance (MaxText)](https://maxtext.readthedocs.io/en/latest/guides/optimization/pallas_kernels_performance.html)

---

Forked from [Triton Puzzles](https://github.com/srush/Triton-Puzzles) · Part of the [puzzle series](https://github.com/srush/gpu-puzzles) by srush et al.
